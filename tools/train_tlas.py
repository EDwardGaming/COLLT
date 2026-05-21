"""
T_LAS — Legal Article Search.

Manuscript §4.1.2 formulates this as standard multi-class classification:
the model takes a user query / case-fact and outputs a softmax over the set
of legal articles in the knowledge base; cross-entropy is the objective.

  Supervised signal (R3#1 fix, 修订版):
    论文初版使用 CAIL2018,与 Table 2 的 LCP/LAP/PTP 测试集存在 id
    重叠,故审稿人 3 判定数据泄漏。改用本地 outputs/tlas_train.jsonl,
    它由 build_tool_train.py 从 DISC-Law-SFT (`judgement_pred` / `exam`
    / `reading_compre`) 抽取得到,与 CAIL2018 test 在 src_id 层面无
    重叠。"CJO" / "PKU" 并非独立数据集,只是 DISC-Law-SFT 部分条目
    的 src_id 前缀标签;这些来源在后续 LawBench 评测中不出现,保留
    无泄漏风险,也无需单独下载或剔除。

Record schema (from build_tool_train.py):
    {"id":..., "src_id":..., "fact":"<case-fact text>",
     "label":"刑法#114",  "all_labels":["刑法#114", ...]}

Training config (matches §4.1.2):
    optimizer AdamW, lr 5e-5, linear decay
    batch_size 16, epochs 10 with early stopping
    balanced sampling to tame the long-tail article distribution.

Usage:
    python -m train.tools.train_tlas
    python -m train.tools.train_tlas --epochs 3 --limit 1000     # smoke
"""
from __future__ import annotations
import argparse, json, os, random
from collections import Counter
from pathlib import Path

# Lawformer-on-Trainer + 多卡 DataParallel 在 transformers ≥ 4.40 会触发
# next(self.parameters()) StopIteration（DP replica 上 parameters() 生成器空）。
# Lawformer 仅 ~110M 参数,单卡 4090 完全够；锁到单卡规避该 bug。
# 必须在 import torch 之前设置才生效。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..common import TLAS_TRAIN, CKPT_DIR, jsonl_iter
from .lawformer_backbone import load_tokenizer, load_for_classification


class TLASDataset(Dataset):
    def __init__(self, records, tokenizer, label2id, max_len=512):
        self.records = records
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):  return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        enc = self.tok(r["fact"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels":         torch.tensor(self.label2id[r["label"]],
                                               dtype=torch.long)}


def _build_label_space(records):
    """Collect every article that appears as a label or in all_labels."""
    labels: set[str] = set()
    for r in records:
        labels.add(r["label"])
        for lab in r.get("all_labels", []) or []:
            labels.add(lab)
    sorted_labels = sorted(labels)
    return {lab: i for i, lab in enumerate(sorted_labels)}


def _balanced_weights(records, label2id):
    """Inverse-frequency weights for WeightedRandomSampler (long-tail fix)."""
    freq = Counter(r["label"] for r in records)
    inv  = {lab: 1.0 / freq[lab] for lab in freq}
    return [inv[r["label"]] for r in records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    type=str, default=str(TLAS_TRAIN))
    p.add_argument("--out",     type=str, default=str(CKPT_DIR / "tool_tlas"))
    p.add_argument("--epochs",  type=int, default=10)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--grad_acc",type=int, default=2)
    p.add_argument("--lr",      type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--val_frac",type=float, default=0.05)
    p.add_argument("--patience",type=int, default=3, help="early-stopping epochs")
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--seed",    type=int, default=3407)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    records = list(jsonl_iter(args.data))
    if args.limit: records = records[:args.limit]
    random.shuffle(records)
    print(f"[data] T_LAS records: {len(records)}")

    label2id = _build_label_space(records)
    id2label = {i: lab for lab, i in label2id.items()}
    print(f"[data] label space size: {len(label2id)}")

    n_val = max(1, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]

    tok = load_tokenizer()
    model = load_for_classification(num_labels=len(label2id))
    model.config.id2label = id2label
    model.config.label2id = label2id

    train_ds = TLASDataset(train, tok, label2id, args.max_len)
    val_ds   = TLASDataset(val,   tok, label2id, args.max_len)

    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

    args_hf = TrainingArguments(
        output_dir            = args.out,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        gradient_accumulation_steps = args.grad_acc,
        gradient_checkpointing      = True,
        num_train_epochs      = args.epochs,
        learning_rate         = args.lr,
        lr_scheduler_type     = "linear",
        warmup_ratio          = 0.06,
        weight_decay          = 0.01,
        eval_strategy         = "epoch",
        save_strategy         = "epoch",
        save_total_limit      = 2,
        load_best_model_at_end= True,
        metric_for_best_model = "accuracy",
        logging_steps         = 25,
        bf16                  = torch.cuda.is_bf16_supported(),
        fp16                  = not torch.cuda.is_bf16_supported(),
        report_to             = "none",
        seed                  = args.seed,
    )

    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    # balanced sampler → swap into Trainer
    weights = torch.tensor(_balanced_weights(train, label2id), dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)

    class _BalancedTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size,
                              sampler=sampler, collate_fn=self.data_collator,
                              num_workers=2, pin_memory=True)

    from transformers import EarlyStoppingCallback
    trainer = _BalancedTrainer(
        model=model, args=args_hf,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=metrics,
        data_collator=DataCollatorWithPadding(tok),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(Path(args.out) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] T_LAS checkpoint → {args.out}")


if __name__ == "__main__":
    main()
