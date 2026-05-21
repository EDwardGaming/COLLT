"""
Unified tool-server façade used by COLLT streaming inference.

Each tool exposes the same callable signature:

    g_t = tool(query: str, history: list[dict] | None = None) -> str

so that the streaming runtime can dispatch on the head-tag name without
caring about the underlying backbone.

Implements the priority order required by reviewer R3#6 (Chinese statutory
law is primary, case law secondary): when both T_LAS and T_SCR are
selected for the same turn, T_LAS *must* be invoked first and its
returned articles are passed to T_SCR as additional context.

To make ablations easy (R3#7), every tool wrapper consults
`os.environ.get("COLLT_ABLATE", "")` — if its short name is listed,
the tool returns the empty string. The runtime additionally rewrites
emitted `<X>g</X>` to `<X></X>` so ablation is fully end-to-end.
"""
from __future__ import annotations
import json, os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import torch

from ..common import CKPT_DIR, TOOL_NAMES


# ── tool priority for R3#6 (Chinese statutory law > case law) ──────────
TOOL_PRIORITY = ["LAS", "LCP", "LER", "LED", "SCR", "NET"]


def ablation_set() -> set[str]:
    raw = os.environ.get("COLLT_ABLATE", "").strip()
    if not raw: return set()
    if raw.upper() == "ALL": return set(TOOL_NAMES)
    return {t.strip().upper() for t in raw.split(",") if t.strip()}


# ───────────────────────── T_LAS ───────────────────────────────────────
@lru_cache(maxsize=1)
def _load_tlas():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root = CKPT_DIR / "tool_tlas"
    tok  = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    mdl  = AutoModelForSequenceClassification.from_pretrained(root, trust_remote_code=True).eval()
    with open(root / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    id2label = {int(k): v for k, v in m["id2label"].items()}
    mdl.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl, id2label


def call_tlas(query: str, history=None, top_k: int = 3) -> str:
    if "LAS" in ablation_set(): return ""
    tok, mdl, id2label = _load_tlas()
    enc = tok(query, return_tensors="pt", truncation=True, max_length=512).to(mdl.device)
    with torch.no_grad():
        logits = mdl(**enc).logits[0]
    top = torch.topk(logits, k=min(top_k, logits.numel())).indices.tolist()
    return "；".join(id2label[i] for i in top)


# ───────────────────────── T_LCP ───────────────────────────────────────
@lru_cache(maxsize=1)
def _load_tlcp():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root = CKPT_DIR / "tool_tlcp"
    tok  = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    mdl  = AutoModelForSequenceClassification.from_pretrained(root, trust_remote_code=True).eval()
    with open(root / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    id2charge = {int(k): v for k, v in m["id2charge"].items()}
    mdl.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl, id2charge


def call_tlcp(query: str, history=None, threshold: float = 0.5) -> str:
    if "LCP" in ablation_set(): return ""
    tok, mdl, id2charge = _load_tlcp()
    enc = tok(query, return_tensors="pt", truncation=True, max_length=512).to(mdl.device)
    with torch.no_grad():
        probs = torch.sigmoid(mdl(**enc).logits[0])
    keep = [(id2charge[i], float(probs[i])) for i in range(probs.numel())
            if probs[i] >= threshold]
    keep.sort(key=lambda x: -x[1])
    if not keep:
        i = int(probs.argmax())
        return id2charge[i]
    return "、".join(c for c, _ in keep)


# ───────────────────────── T_SCR ───────────────────────────────────────
@lru_cache(maxsize=1)
def _load_tscr():
    from transformers import AutoTokenizer
    from .lawformer_backbone import load_for_encoding
    from .train_tscr import SiameseLawformer
    root = CKPT_DIR / "tool_tscr"
    tok  = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    backbone = load_for_encoding()
    mdl = SiameseLawformer(backbone)
    state = torch.load(root / "siamese.pt", map_location="cpu")
    mdl.load_state_dict(state["state"])
    mdl.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # case bank: load from JSONL on disk; users prepare this offline
    bank_path = root / "case_bank.jsonl"
    bank: list[dict] = []
    if bank_path.exists():
        import orjson
        with open(bank_path, "rb") as f:
            for line in f:
                if line.strip():
                    bank.append(orjson.loads(line))

    # pre-encode bank
    bank_vecs = None
    if bank:
        ids, masks = [], []
        for r in bank:
            e = tok(r["text"], truncation=True, max_length=512,
                    padding="max_length", return_tensors="pt")
            ids.append(e["input_ids"][0]); masks.append(e["attention_mask"][0])
        ids = torch.stack(ids).to(mdl.backbone.device)
        masks = torch.stack(masks).to(mdl.backbone.device)
        with torch.no_grad():
            bank_vecs = mdl.encode(ids, masks).cpu()
    return tok, mdl, bank, bank_vecs


def call_tscr(query: str, history=None, top_k: int = 1) -> str:
    if "SCR" in ablation_set(): return ""
    tok, mdl, bank, bank_vecs = _load_tscr()
    if bank_vecs is None or not bank:
        return ""   # no case bank prepared yet
    enc = tok(query, return_tensors="pt", truncation=True, max_length=512,
              padding="max_length").to(mdl.backbone.device)
    with torch.no_grad():
        q_vec = mdl.encode(enc["input_ids"], enc["attention_mask"]).cpu()
    sims = torch.nn.functional.cosine_similarity(q_vec, bank_vecs)
    idx = sims.topk(min(top_k, sims.numel())).indices.tolist()
    return "；".join(bank[i]["text"][:300] for i in idx)


# ───────────────────────── T_LER ───────────────────────────────────────
# 重构说明:本工具按论文 §4.1.4 表述「事实描述句子 → 0~多个元素标签」
# 实现为句子级 multi-label classification(CAIL2019 element-extraction
# 的原生格式)。旧版按 BIO token-cls 的实现已删除。
import re as _re

@lru_cache(maxsize=1)
def _load_tler():
    """Returns (tokenizer, model, id2label, label2name) or None if absent."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root = CKPT_DIR / "tool_tler"
    if not (root / "config.json").exists():
        return None
    tok = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        str(root), trust_remote_code=True).eval()
    with open(root / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    id2label   = {int(k): v for k, v in m["id2label"].items()}
    label2name = m.get("label2name", {}) or {}
    mdl.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl, id2label, label2name


_SENT_SPLIT = _re.compile(r"[^。！？\?!\n]+[。！？\?!]?")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.findall(text) if s.strip()]


def call_tler(query: str, history=None, threshold: float = 0.5) -> str:
    if "LER" in ablation_set(): return ""
    result = _load_tler()
    if result is None:
        return ""   # checkpoint not trained — T_LER gracefully disabled
    tok, mdl, id2label, label2name = result

    # 切句,逐句多标签预测,再聚合
    sents = _split_sentences(query) or [query.strip()]
    out_pairs: list[tuple[str, str]] = []   # (sentence_preview, label_name)
    seen_labels: set[str] = set()
    with torch.no_grad():
        for sent in sents:
            enc = tok(sent, return_tensors="pt", truncation=True,
                      max_length=256, padding=True).to(mdl.device)
            probs = torch.sigmoid(mdl(**enc).logits[0])
            for i in range(probs.numel()):
                if float(probs[i]) >= threshold:
                    tag = id2label[i]
                    if tag in seen_labels:
                        continue
                    seen_labels.add(tag)
                    name = label2name.get(tag, tag)
                    preview = sent[:30] + ("…" if len(sent) > 30 else "")
                    out_pairs.append((preview, name))
    return "；".join(f"{s}({n})" for s, n in out_pairs)


# ───────────────────────── T_LED ───────────────────────────────────────
@lru_cache(maxsize=1)
def _load_tled():
    """Returns (tokenizer, model) or None if checkpoint is absent.

    Trained by train_tled.py on local_data/LEVEN/.
    Returns None gracefully when no checkpoint exists.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    root = CKPT_DIR / "tool_tled"
    if not (root / "config.json").exists():
        return None
    tok = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    mdl = AutoModelForTokenClassification.from_pretrained(
        str(root), trust_remote_code=True).eval()
    mdl.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl


def call_tled(query: str, history=None) -> str:
    if "LED" in ablation_set(): return ""
    result = _load_tled()
    if result is None:
        return ""   # checkpoint not trained — T_LED gracefully disabled
    tok, mdl = result
    enc = tok(query, return_tensors="pt", truncation=True, max_length=512,
              return_offsets_mapping=True).to(mdl.device)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        pred = mdl(**enc).logits[0].argmax(-1).tolist()
    id2label = mdl.config.id2label
    events: list[str] = []
    for tag_id, (a, b) in zip(pred, offsets):
        tag = id2label[tag_id]
        if tag != "O" and a < b:
            events.append(f"{query[a:b]}:{tag}")
    return "；".join(events)


# ───────────────────────── T_NET (Bing Web Search) ─────────────────────
def call_tnet(query: str, history=None, top_k: int = 3) -> str:
    if "NET" in ablation_set(): return ""
    import urllib.request, urllib.parse
    api_key = os.environ.get("BING_API_KEY")
    if not api_key:
        return ""
    url = ("https://api.bing.microsoft.com/v7.0/search?"
           + urllib.parse.urlencode({"q": query, "count": top_k, "mkt": "zh-CN"}))
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception:
        return ""
    snippets = [it.get("snippet", "") for it in data.get("webPages", {}).get("value", [])]
    return "；".join(s for s in snippets if s)


# ───────────────────────── dispatcher table ────────────────────────────
TOOL_FN: dict[str, Callable[..., str]] = {
    "SCR": call_tscr,
    "LAS": call_tlas,
    "LCP": call_tlcp,
    "LER": call_tler,
    "LED": call_tled,
    "NET": call_tnet,
}


def dispatch(name: str, query: str,
             history: Optional[list[dict]] = None,
             context: Optional[dict] = None) -> str:
    """Invoke a single tool by short name. `context` may carry results from
    higher-priority tools so we can feed e.g. T_LAS articles into T_SCR."""
    name = name.upper()
    fn = TOOL_FN.get(name)
    if fn is None:
        return ""
    enriched = query
    if context and name == "SCR" and "LAS" in context and context["LAS"]:
        enriched = f"{query}\n\n[已检索到的法条]\n{context['LAS']}"
    return fn(enriched, history)


def order_by_priority(tools: list[str]) -> list[str]:
    """Re-order a multi-tool selection per R3#6 priority (statute > case)."""
    rank = {t: i for i, t in enumerate(TOOL_PRIORITY)}
    return sorted(tools, key=lambda t: rank.get(t.upper(), 99))
