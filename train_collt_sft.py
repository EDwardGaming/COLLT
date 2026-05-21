"""
COLLT 指令微调主入口(unsloth + 4-bit QLoRA,RTX 4090 24G 可单卡训练)。

实现论文 §3.1.4 / Eq.12 的联合损失:

    L_total = L_action + α·L_tool + β·L_resp ,  α = β = 1

由于 <CLR>/<DRT> 动作 token、工具 <XXX></XXX> 标签、解析结果 g_t、以及
<ER> 后的增强答复都是 assistant 段的 *输出 token*,在 assistant span 上
做 token-level NLL 就严格等价于上式,无需额外辅助头。

四张表:Table 3 列出的 5 个基座(--model)
    glm | llama | internlm | qwen | baichuan

使用:
    python -m train.train_collt_sft --model qwen          # 默认参数即 4090 可跑
    python -m train.train_collt_sft --model llama --epochs 2
    python -m train.train_collt_sft --model qwen --limit 200    # 冒烟

4090 24G 单卡显存预算:
    7B 4-bit base + LoRA r=16 ≈ 5 GB
    grad checkpointing on, batch=1, grad_acc=16 → 等效 batch 16
    max_seq_len=1536 平均长度足够覆盖 collt_sft.jsonl 的多轮对话
    paged_adamw_8bit 让优化器状态从显存下放
全套压在 ~20 GB 内,留 4 GB 余量给 logits / token embedding 扩表。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from .common import COLLT_SFT, CKPT_DIR, SPECIAL_TOKENS, base_model_id
from .data_collt import build_chat_dataset


# 每个基座模型的多轮分隔符 — 来自其官方 chat_template。
# train_on_responses_only 依据这两段 marker 切出 assistant span 做 loss-mask;
# 若与实际 chat_template 不匹配,会导致 loss mask 失效(全 mask 或全不 mask)。
CHAT_DELIMS: dict[str, tuple[str, str]] = {
    # ChatML 系列(Qwen2/Qwen2.5、InternLM2/3 都是 ChatML)
    "qwen":     ("<|im_start|>user\n",          "<|im_start|>assistant\n"),
    "internlm": ("<|im_start|>user\n",          "<|im_start|>assistant\n"),
    # LLaMa 3 Instruct
    "llama":    ("<|start_header_id|>user<|end_header_id|>\n\n",
                 "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    # ChatGLM3
    "glm":      ("<|user|>\n",                  "<|assistant|>\n"),
    # Baichuan2 chat 用预留 token 做角色分隔
    "baichuan": ("<reserved_106>",              "<reserved_107>"),
}


def _load_unsloth(model_id: str, max_len: int, dtype: str, load_in_4bit: bool):
    """延迟 import,避免不需要 unsloth 时本文件依然可被解析。"""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = model_id,
        max_seq_length  = max_len,
        dtype           = None if dtype == "auto" else dtype,
        load_in_4bit    = load_in_4bit,
        trust_remote_code = True,         # ChatGLM3 / InternLM / Baichuan 都需要
    )
    return model, tokenizer


def _attach_lora(model, r: int, alpha: int, dropout: float):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r              = r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["glm", "llama", "internlm", "qwen", "baichuan"])
    parser.add_argument("--data",      type=str, default=str(COLLT_SFT))
    parser.add_argument("--out",       type=str, default=None)
    # === 4090-friendly defaults ===
    parser.add_argument("--max_len",   type=int, default=1536)
    parser.add_argument("--epochs",    type=int, default=3)
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--grad_acc",  type=int, default=16)
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--lora_r",    type=int, default=16)
    parser.add_argument("--lora_alpha",type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_4bit",   action="store_true",
                        help="关掉 4-bit 量化(显存足够时用)")
    parser.add_argument("--dtype",     default="auto", choices=["auto","bf16","fp16"])
    parser.add_argument("--limit",     type=int, default=None,
                        help="冒烟测试用,截断数据集到 N 行")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else CKPT_DIR / f"collt-{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────
    ds = build_chat_dataset(args.data, limit=args.limit)
    print(f"[data] {len(ds)} dialogues from {args.data}")

    # ── model + tokenizer ─────────────────────────────────────────────
    model, tokenizer = _load_unsloth(
        base_model_id(args.model),
        args.max_len, args.dtype,
        load_in_4bit=not args.no_4bit,
    )

    # Qwen 系 / LLaMa3 / ChatGLM 部分基座原生没有 pad_token,unsloth 会现场
    # 新建 <|PAD_TOKEN|>,但这会和后面 add_special_tokens + resize 的顺序冲突
    # (新 pad 行 embedding 没有初始化)。规范做法:用 eos_token 复用已有 embedding。
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        print(f"[tokenizer] pad_token ← eos_token = {tokenizer.eos_token!r}")

    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
        # 新增 token 的 embedding 用已有词嵌入均值初始化,比随机分布更稳定。
        with __import__("torch").no_grad():
            emb = model.get_input_embeddings().weight
            mean_vec = emb[: -n_added].mean(dim=0)
            emb[-n_added:] = mean_vec.unsqueeze(0).expand(n_added, -1).clone()
            out = model.get_output_embeddings()
            if out is not None and out.weight.shape[0] == emb.shape[0]:
                mean_out = out.weight[: -n_added].mean(dim=0)
                out.weight[-n_added:] = mean_out.unsqueeze(0).expand(n_added, -1).clone()
        print(f"[tokenizer] 新增特殊 token: {n_added} 个 "
              f"(<CLR>/<DRT>/<XXX>…</XXX>/<ER>),embedding 用均值初始化")

    model = _attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # 每行渲染一次 chat 模板,SFTTrainer 看到的就是纯文本字段 "text"
    def _format(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False,
                                                       add_generation_prompt=False)
                         for m in batch["messages"]]}
    ds = ds.map(_format, batched=True, remove_columns=ds.column_names)

    # ── trainer ───────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    sft_cfg = SFTConfig(
        output_dir            = str(out_dir),
        per_device_train_batch_size = args.batch,
        gradient_accumulation_steps = args.grad_acc,
        num_train_epochs      = args.epochs,
        learning_rate         = args.lr,
        warmup_ratio          = 0.03,
        lr_scheduler_type     = "cosine",
        weight_decay          = 0.01,
        logging_steps         = 10,
        save_strategy         = "epoch",
        save_total_limit      = 2,
        bf16                  = (args.dtype != "fp16"),
        fp16                  = (args.dtype == "fp16"),
        optim                 = "paged_adamw_8bit",  # 4090 友好
        max_seq_length        = args.max_len,
        dataset_text_field    = "text",
        packing               = False,
        seed                  = 3407,
        gradient_checkpointing= True,
        report_to             = "none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=ds, args=sft_cfg)
    # mask 掉非 assistant 段,token-level NLL 就严格落在 L_action+L_tool+L_resp 上。
    # 不同基座的 chat_template 不一样,marker 必须按模型选,否则多轮 mask 会失配。
    if args.model not in CHAT_DELIMS:
        raise ValueError(
            f"未知基座 {args.model};请在 CHAT_DELIMS 里登记其 chat_template "
            "中 user / assistant 段的起始字串。"
        )
    inst_part, resp_part = CHAT_DELIMS[args.model]
    print(f"[sft] response-only masking markers for {args.model}: "
          f"user={inst_part!r} assistant={resp_part!r}")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = inst_part,
        response_part    = resp_part,
    )

    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    print(f"[done] checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
