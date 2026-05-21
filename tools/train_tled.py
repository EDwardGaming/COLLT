"""
T_LED — Legal Event Detection.

Fine-tunes a Lawformer token-classification head on the LEVEN dataset to
identify and classify legal events (contract signing, tortious conduct,
intentional injury, …) from case-fact text.

Training data (local_data/LEVEN/):
  train.jsonl  — 5301 case documents
  valid.jsonl  — 1230 case documents

LEVEN document format (per line):
  {
    "id": ..., "crime": ...,
    "content": [
      {"sentence": "...", "tokens": ["tok1", "tok2", ...]},
      ...
    ],
    "events": [
      {"type": "诈骗", "type_id": 71,
       "mention": [{"trigger_word": "诈骗", "sent_id": 0,
                    "offset": [17, 18], ...}]},
      ...
    ]
  }

Conversion to per-sentence BIO:
  For each sentence, mark trigger tokens as B-{type}/I-{type}; rest as O.
  All triggers in LEVEN are single-token (offset span = 1 token wide),
  so I- labels do not appear in practice but are kept for completeness.

Usage:
    python -m train.tools.train_tled
    python -m train.tools.train_tled --train path/to/train.jsonl
    python -m train.tools.train_tled --limit 200   # smoke test
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import Iterator

# 同 train_tlas.py：DataParallel + Longformer.dtype 已知冲突,锁单卡规避。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

from ..common import CKPT_DIR, LEVEN_DIR
from .lawformer_backbone import LAWFORMER_ID


# ── LEVEN → BIO conversion ────────────────────────────────────────────────────

def _iter_bio(path: Path, limit: int | None = None) -> Iterator[dict]:
    """Yield per-sentence {"tokens": [...], "labels": [...]} from LEVEN jsonl."""
    with open(path, encoding="utf-8") as f:
        for doc_idx, line in enumerate(f):
            if limit and doc_idx >= limit:
                break
            doc = json.loads(line)
            # build per-sentence, per-token event index
            # {sent_id: {tok_pos: event_type}}
            ev_map: dict[int, dict[int, str]] = {}
            for ev in doc.get("events", []):
                etype = ev["type"]
                for mention in ev["mention"]:
                    sid = mention["sent_id"]
                    start, end = mention["offset"]
                    if sid not in ev_map:
                        ev_map[sid] = {}
                    ev_map[sid][start] = f"B-{etype}"
                    for j in range(start + 1, end):
                        ev_map[sid][j] = f"I-{etype}"

            for sid, sent in enumerate(doc.get("content", [])):
                tokens = sent["tokens"]
                if not tokens:
                    continue
                sent_ev = ev_map.get(sid, {})
                labels = [sent_ev.get(j, "O") for j in range(len(tokens))]
                yield {"tokens": tokens, "labels": labels}


def _collect_label_space(paths: list[Path]) -> list[str]:
    """Collect all unique BIO tags across files."""
    tags: set[str] = {"O"}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                for ev in doc.get("events", []):
                    t = ev["type"]
                    tags.add(f"B-{t}")
                    tags.add(f"I-{t}")
    return sorted(tags)


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train",   type=str, default=str(LEVEN_DIR / "train.jsonl"),
                   help="LEVEN train.jsonl path")
    p.add_argument("--valid",   type=str, default=str(LEVEN_DIR / "valid.jsonl"),
                   help="LEVEN valid.jsonl path (used as eval set)")
    p.add_argument("--out",     default=str(CKPT_DIR / "tool_tled"))
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--grad_acc",type=int, default=2)
    p.add_argument("--lr",      type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--limit",   type=int, default=None,
                   help="Limit to N documents for smoke testing")
    p.add_argument("--seed",    type=int, default=3407)
    args = p.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)

    if not train_path.exists():
        import sys
        print(
            f"[T_LED] Train file not found: {train_path}\n"
            f"        Expected at: {LEVEN_DIR / 'train.jsonl'}\n"
            "        Download LEVEN from:\n"
            "          https://thunlp.oss-cn-qingdao.aliyuncs.com/LEVEN/LEVEN.zip\n"
            "        and place train.jsonl / valid.jsonl under local_data/LEVEN/.",
            file=sys.stderr,
        )
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # collect label space from both splits (ensures consistent IDs)
    label_paths = [train_path]
    if valid_path.exists():
        label_paths.append(valid_path)
    print("[T_LED] Building label space …")
    label_list = _collect_label_space(label_paths)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label  = {i: l for l, i in label2id.items()}
    print(f"[data] LED tag space ({len(label_list)}): {label_list[:8]} …")

    # convert to sentence-level records
    train_recs = list(_iter_bio(train_path, args.limit))
    val_recs   = list(_iter_bio(valid_path, args.limit)) if valid_path.exists() else []
    print(f"[data] train sentences: {len(train_recs)},  val sentences: {len(val_recs)}")

    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              TrainingArguments, Trainer,
                              DataCollatorForTokenClassification)
    from datasets import Dataset as HFDataset

    tok = AutoTokenizer.from_pretrained(LAWFORMER_ID, trust_remote_code=True)

    def tokenize_and_align(example):
        toks = tok(example["tokens"], is_split_into_words=True,
                   truncation=True, max_length=args.max_len)
        aligned = []
        word_ids = toks.word_ids()
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(label2id[example["labels"][wid]])
            else:
                aligned.append(-100)
            prev = wid
        toks["labels"] = aligned
        return toks

    col_remove = ["tokens", "labels"]
    train_ds = HFDataset.from_list(train_recs).map(
        tokenize_and_align, remove_columns=col_remove)
    eval_ds  = HFDataset.from_list(val_recs).map(
        tokenize_and_align, remove_columns=col_remove) if val_recs else None

    model = AutoModelForTokenClassification.from_pretrained(
        LAWFORMER_ID, num_labels=len(label_list),
        id2label=id2label, label2id=label2id,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    )

    targs = TrainingArguments(
        output_dir                  = args.out,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        gradient_accumulation_steps = args.grad_acc,
        gradient_checkpointing      = True,
        num_train_epochs            = args.epochs,
        learning_rate               = args.lr,
        weight_decay                = 0.01,
        eval_strategy               = "epoch" if eval_ds else "no",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = bool(eval_ds),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
        seed                        = args.seed,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tok),
    )
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(Path(args.out) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] T_LED → {args.out}")


if __name__ == "__main__":
    main()
