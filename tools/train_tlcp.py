"""
T_LCP — Legal Charge Prediction (multi-label).

Manuscript §4.1.3 formulates this as multi-label classification over charges
with binary cross-entropy:

    L = − Σ_i [ y_i log p_i + (1 − y_i) log(1 − p_i) ]

  Supervised signal (R3#1 fix, 修订版):
    审稿人 3 指出 T_LCP 原版训练集与 CAIL2018 评测集重叠造成泄漏。
    改用 outputs/tlcp_train.jsonl,由 build_tool_train.py 从
    DISC-Law-SFT (`sent_pred` / `judgement_pred` / `case_class`)
    抽取,在 src_id 层面与 CAIL2018 test 不重叠。
    "CJO" / "PKU" 并非独立数据集,只是 DISC-Law-SFT 部分条目的
    src_id 前缀标签;后续 LawBench 评测不涉及这些来源,保留无泄漏
    风险,也无需单独下载或剔除。

Record schema:
    {"id":..., "src_id":..., "fact":"<case-fact text>",
     "charges":["放火罪", "盗窃罪", ...]}

Training config (matches §4.1.3):
    Adam, lr 2e-5, batch 32, max_len 512,
    dropout 0.3, L2 (weight_decay) 0.01,
    epochs 5 with F1-based early stopping (patience 3).
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

# 同 train_tlas.py：DataParallel + Longformer.dtype 已知冲突,锁单卡规避。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import TLCP_TRAIN, CKPT_DIR, jsonl_iter
from .lawformer_backbone import load_tokenizer, load_for_classification


class TLCPDataset(Dataset):
    def __init__(self, records, tokenizer, charge2id, max_len=512):
        self.records = records
        self.tok = tokenizer
        self.charge2id = charge2id
        self.max_len = max_len
        self.n = len(charge2id)

    def __len__(self):  return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        enc = self.tok(r["fact"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        y = torch.zeros(self.n, dtype=torch.float)
        for c in r["charges"]:
            if c in self.charge2id:
                y[self.charge2id[c]] = 1.0
        return {"input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels":         y}


def _build_label_space(records):
    charges: set[str] = set()
    for r in records:
        charges.update(r["charges"])
    sorted_c = sorted(charges)
    return {c: i for i, c in enumerate(sorted_c)}


def _multilabel_f1(probs: np.ndarray, gold: np.ndarray, thr: float = 0.5) -> float:
    pred = (probs >= thr).astype(int)
    tp = ((pred == 1) & (gold == 1)).sum()
    fp = ((pred == 1) & (gold == 0)).sum()
    fn = ((pred == 0) & (gold == 1)).sum()
    if tp + fp == 0 or tp + fn == 0: return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    type=str, default=str(TLCP_TRAIN))
    p.add_argument("--out",     type=str, default=str(CKPT_DIR / "tool_tlcp"))
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--grad_acc",type=int, default=4)
    p.add_argument("--lr",      type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--val_frac",type=float, default=0.05)
    p.add_argument("--patience",type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--seed",    type=int, default=3407)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    records = list(jsonl_iter(args.data))
    if args.limit: records = records[:args.limit]
    random.shuffle(records)
    print(f"[data] T_LCP records: {len(records)}")

    c2i = _build_label_space(records)
    i2c = {i: c for c, i in c2i.items()}
    print(f"[data] charge space size: {len(c2i)}")

    n_val = max(1, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]

    tok = load_tokenizer()
    model = load_for_classification(num_labels=len(c2i),
                                    problem_type="multi_label_classification")
    # boost dropout per §4.1.3 (0.3)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout

    model.config.id2label = i2c
    model.config.label2id = c2i

    train_ds = TLCPDataset(train, tok, c2i, args.max_len)
    val_ds   = TLCPDataset(val,   tok, c2i, args.max_len)

    from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding,
                              EarlyStoppingCallback)

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
        weight_decay          = 0.01,           # L2 regularization
        eval_strategy         = "epoch",
        save_strategy         = "epoch",
        save_total_limit      = 2,
        load_best_model_at_end= True,
        metric_for_best_model = "f1",
        greater_is_better     = True,
        logging_steps         = 25,
        bf16                  = torch.cuda.is_bf16_supported(),
        fp16                  = not torch.cuda.is_bf16_supported(),
        report_to             = "none",
        seed                  = args.seed,
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        return {"f1": _multilabel_f1(probs, labels)}

    trainer = Trainer(
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
        json.dump({"charge2id": c2i, "id2charge": i2c},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] T_LCP checkpoint → {args.out}")


if __name__ == "__main__":
    main()
