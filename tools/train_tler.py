"""
T_LER — Legal Element Recognition (sentence-level multi-label classification).

数据来源(对应论文 §4.1.4 中的「基于 CAIL2019 收集的离婚数据集」):

  CAIL2019 element-extraction (china-ai-law-challenge/CAIL2019)。
  原比赛把每个判决书分成若干句子,逐句给 0~多个元素标签(divorce 域
  DV1–DV20、labor 域 LB1–LB20、loan 域 LN1–LN20)。这一格式天然就是
  **句子级多标签分类**,与论文 §4.1.4 表述完全一致(原版 train_tler.py
  按 BIO token-classification 实现,与本数据集结构不符,已重写)。

  数据由 build_ler_data.py 拉取并平铺到:
    train/outputs/tler_train.jsonl   (来自 train_selected.json)
    train/outputs/tler_valid.jsonl   (来自 data_small_selected.json)
  每行:  {"sentence": "...", "labels": ["DV1", ...], "domain": "divorce"}

训练配置:
  Lawformer + multi-label sigmoid head,batch 8, lr 3e-5, max_len 256
  (单句通常远短于 512), epoch 5, weight_decay 0.01, dropout 0.2,
  F1 早停。

用法:
    # 一键:先构数据再训练
    python -m train.tools.build_ler_data
    python -m train.tools.train_tler

    # 冒烟:
    python -m train.tools.train_tler --limit 1000 --epochs 1
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

# 同 train_tlas.py：DataParallel + Longformer.dtype 已知冲突,锁单卡规避。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import (CKPT_DIR, OUT_DIR, jsonl_iter,
                      TLER_TRAIN, TLER_VALID, TLER_LABEL_NAMES)
from .lawformer_backbone import load_tokenizer, load_for_classification


class TLERDataset(Dataset):
    def __init__(self, records, tokenizer, label2id, max_len: int):
        self.records = records
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.n = len(label2id)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        enc = self.tok(r["sentence"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        y = torch.zeros(self.n, dtype=torch.float)
        for lab in r["labels"]:
            if lab in self.label2id:
                y[self.label2id[lab]] = 1.0
        return {"input_ids":      enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels":         y}


def _build_label_space(records) -> dict[str, int]:
    labels: set[str] = set()
    for r in records:
        labels.update(r["labels"])
    return {lab: i for i, lab in enumerate(sorted(labels))}


def _multilabel_f1(probs: np.ndarray, gold: np.ndarray, thr: float = 0.5) -> float:
    pred = (probs >= thr).astype(int)
    tp = ((pred == 1) & (gold == 1)).sum()
    fp = ((pred == 1) & (gold == 0)).sum()
    fn = ((pred == 0) & (gold == 1)).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train",   default=str(TLER_TRAIN))
    p.add_argument("--valid",   default=str(TLER_VALID))
    p.add_argument("--out",     default=str(CKPT_DIR / "tool_tler"))
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--grad_acc",type=int, default=2)
    p.add_argument("--lr",      type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--patience",type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--limit",   type=int, default=None,
                   help="截断训练集到 N 条句子(冒烟测试用)")
    p.add_argument("--seed",    type=int, default=3407)
    args = p.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    if not train_path.exists() or not valid_path.exists():
        print(
            f"[T_LER] 缺少训练 / 验证文件,先运行:\n"
            f"          python -m train.tools.build_ler_data\n"
            f"        以下载 CAIL2019 元素识别数据并生成:\n"
            f"          {train_path}\n          {valid_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_recs = list(jsonl_iter(train_path))
    val_recs   = list(jsonl_iter(valid_path))
    if args.limit:
        train_recs = train_recs[:args.limit]
    random.shuffle(train_recs)
    print(f"[data] T_LER 训练句={len(train_recs)}  验证句={len(val_recs)}")

    label2id = _build_label_space(train_recs + val_recs)
    id2label = {i: lab for lab, i in label2id.items()}
    print(f"[data] 标签空间大小: {len(label2id)} ({sorted(label2id)[:6]} …)")

    # 拼上中文名映射 (来自 selectedtags.txt) 供推理时人类可读
    label2name: dict[str, str] = {}
    if TLER_LABEL_NAMES.exists():
        with open(TLER_LABEL_NAMES, encoding="utf-8") as f:
            label2name = json.load(f)

    tok = load_tokenizer()
    model = load_for_classification(num_labels=len(label2id),
                                    problem_type="multi_label_classification")
    # §4.1.4 配套的 dropout 提升
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout
    model.config.id2label = id2label
    model.config.label2id = label2id

    train_ds = TLERDataset(train_recs, tok, label2id, args.max_len)
    val_ds   = TLERDataset(val_recs,   tok, label2id, args.max_len)

    from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding,
                              EarlyStoppingCallback)

    targs = TrainingArguments(
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
        metric_for_best_model = "f1",
        greater_is_better     = True,
        logging_steps         = 50,
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
        model=model, args=targs,
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
        json.dump({"label2id":   label2id,
                   "id2label":   id2label,
                   "label2name": label2name},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] T_LER checkpoint → {args.out}")


if __name__ == "__main__":
    main()
