"""
T_SCR — Similar Case Retrieval (Lawformer + Siamese, manuscript §4.1.1).

Loss = λ1·L_contrastive + λ2·L_triplet, with
  L_contrastive = y·D² + (1−y)·max(0, m − D)²
  L_triplet     = max(0, ‖h_a − h_p‖² − ‖h_a − h_n‖² + η)

Training data:
  local_data/CAIL2019/相似案例匹配 — jsonl format
    {"A": <fact_text>, "B": <fact_text>, "C": <fact_text>, "label": "B"|"C"}
  "label" indicates which of B/C is more similar to A.
  Each row yields one contrastive pair + one triplet.

  (LeCaRDv2 candidate docs are not bundled; LEVEN/CAIL-LeRec are also
   unavailable — this single source is sufficient for the siamese objective.)

Usage:
    python -m train.tools.train_tscr
    python -m train.tools.train_tscr --data path/to/train.json --limit 500
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..common import CKPT_DIR, CAIL2019_SCM_DIR
from .lawformer_backbone import load_tokenizer, load_for_encoding


class SiameseLawformer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dim = backbone.config.hidden_size

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)


class PairDataset(Dataset):
    """y ∈ {0,1} for the contrastive arm."""
    def __init__(self, pairs, tok, max_len=512):
        self.pairs, self.tok, self.max_len = pairs, tok, max_len

    def __len__(self):  return len(self.pairs)

    def _enc(self, t):
        return self.tok(t, truncation=True, max_length=self.max_len,
                        padding="max_length", return_tensors="pt")

    def __getitem__(self, i):
        x1, x2, y = self.pairs[i]
        a, b = self._enc(x1), self._enc(x2)
        return {"a_ids":a["input_ids"][0], "a_mask":a["attention_mask"][0],
                "b_ids":b["input_ids"][0], "b_mask":b["attention_mask"][0],
                "y": torch.tensor(float(y))}


class TripletDataset(Dataset):
    """(anchor, positive, negative) triplets for the triplet arm."""
    def __init__(self, triplets, tok, max_len=512):
        self.triplets, self.tok, self.max_len = triplets, tok, max_len

    def __len__(self):  return len(self.triplets)

    def _enc(self, t):
        return self.tok(t, truncation=True, max_length=self.max_len,
                        padding="max_length", return_tensors="pt")

    def __getitem__(self, i):
        a, p, n = self.triplets[i]
        ea, ep, en = self._enc(a), self._enc(p), self._enc(n)
        return {"a_ids":ea["input_ids"][0], "a_mask":ea["attention_mask"][0],
                "p_ids":ep["input_ids"][0], "p_mask":ep["attention_mask"][0],
                "n_ids":en["input_ids"][0], "n_mask":en["attention_mask"][0]}


def _load_scm_local(path: Path, limit: int | None = None):
    """Load CAIL2019 相似案例匹配 JSONL.

    Each line: {"A": text, "B": text, "C": text, "label": "B"|"C"}
    "label" == "B" → B is more similar to A (positive), C is negative.
    "label" == "C" → C is more similar to A (positive), B is negative.

    Returns (pairs, triplets) where:
      pairs    = [(text1, text2, y)] with y ∈ {0, 1}
      triplets = [(anchor, positive, negative)]
    """
    pairs: list[tuple[str, str, int]] = []
    triplets: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a, b, c, lbl = r["A"], r["B"], r["C"], r["label"]
            if lbl == "B":
                pos, neg = b, c
            else:
                pos, neg = c, b
            pairs.append((a, pos, 1))
            pairs.append((a, neg, 0))
            triplets.append((a, pos, neg))
    return pairs, triplets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   type=str,
                   default=str(CAIL2019_SCM_DIR / "train.json"),
                   help="CAIL2019 SCM train jsonl path")
    p.add_argument("--out",    type=str, default=str(CKPT_DIR / "tool_tscr"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch",  type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr",     type=float, default=5e-5)
    p.add_argument("--margin_contrastive", type=float, default=1.0)
    p.add_argument("--margin_triplet",     type=float, default=0.5)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--limit",  type=int, default=None,
                   help="Truncate input rows for smoke testing")
    p.add_argument("--seed",   type=int, default=3407)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"CAIL2019 SCM data not found: {data_path}\n"
            f"Expected at: {CAIL2019_SCM_DIR / 'train.json'}\n"
            "Make sure local_data/CAIL2019/相似案例匹配/train.json exists."
        )

    pairs, triplets = _load_scm_local(data_path, limit=args.limit)
    print(f"[data] raw pairs={len(pairs)}  triplets={len(triplets)}")

    # 1:3 pos:neg balance for contrastive pairs
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    if len(neg) < 3 * len(pos):
        neg = neg * (3 * len(pos) // max(1, len(neg)) + 1)
    balanced = pos + neg[:3 * len(pos)]
    random.shuffle(balanced)
    print(f"[data] balanced pairs={len(balanced)}  triplets={len(triplets)}")

    tok      = load_tokenizer()
    backbone = load_for_encoding()
    model    = SiameseLawformer(backbone).cuda()

    pair_dl    = DataLoader(PairDataset(balanced, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2)
    triplet_dl = DataLoader(TripletDataset(triplets, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = (len(pair_dl) + len(triplet_dl)) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                              end_factor=0.0, total_iters=total)
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp and not torch.cuda.is_bf16_supported())
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    best, bad = float("inf"), 0
    for ep in range(args.epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        it_p = iter(pair_dl); it_t = iter(triplet_dl)
        accum_step = 0
        opt.zero_grad()
        while True:
            try: bp = next(it_p)
            except StopIteration: bp = None
            try: bt = next(it_t)
            except StopIteration: bt = None
            if bp is None and bt is None:
                break

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                loss = torch.tensor(0.0, device="cuda")
                if bp is not None:
                    bp = {k: v.cuda() for k, v in bp.items()}
                    h1 = model.encode(bp["a_ids"], bp["a_mask"])
                    h2 = model.encode(bp["b_ids"], bp["b_mask"])
                    d  = F.pairwise_distance(h1, h2)
                    y  = bp["y"]
                    lc = y * d.pow(2) + (1 - y) * F.relu(args.margin_contrastive - d).pow(2)
                    loss = loss + args.lambda1 * lc.mean()
                if bt is not None:
                    bt = {k: v.cuda() for k, v in bt.items()}
                    ha = model.encode(bt["a_ids"], bt["a_mask"])
                    hp = model.encode(bt["p_ids"], bt["p_mask"])
                    hn = model.encode(bt["n_ids"], bt["n_mask"])
                    lt = F.relu((ha - hp).pow(2).sum(-1)
                                - (ha - hn).pow(2).sum(-1)
                                + args.margin_triplet)
                    loss = loss + args.lambda2 * lt.mean()
                loss = loss / max(1, args.grad_acc)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_step += 1
            if accum_step % args.grad_acc == 0:
                if scaler.is_enabled():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad()
            loss_sum += loss.item() * args.grad_acc
            n_batches += 1

        mean = loss_sum / max(1, n_batches)
        print(f"[ep {ep:02d}] loss={mean:.4f}")
        if mean < best - 1e-4:
            best, bad = mean, 0
            Path(args.out).mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict()}, Path(args.out) / "siamese.pt")
            tok.save_pretrained(args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] no improvement for {bad} epochs")
                break
    print(f"[done] T_SCR siamese → {args.out}")


if __name__ == "__main__":
    main()
