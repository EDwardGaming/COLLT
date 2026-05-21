"""
Turn outputs/collt_sft.jsonl into an HF Dataset suitable for chat-style SFT.

Each record in collt_sft.jsonl already has the canonical multi-turn shape:

    {"messages":[{"role":"user","content":...},
                 {"role":"assistant","content":"<CLR>..."},
                 {"role":"user","content":...},
                 {"role":"assistant","content":"<DRT><LAS>...</LAS>..."}]}

Assistant content already embeds the action token (<CLR> or <DRT>), all tool
head/tail tags, the tool parsing results g_t, and the <ER>-enhanced response.
The single token-level NLL on assistant spans therefore *is* the multi-task
loss L_total = L_action + α L_tool + β L_resp with α=β=1 (manuscript §3.1.4 /
Equation 12) — no auxiliary heads needed because every choice is realised as
output tokens.

We expose two builders:
    build_chat_dataset(...)   — for unsloth / SFTTrainer (text + assistant mask)
    build_pretokenized(...)   — already-tokenized variant for manual training
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from datasets import Dataset

from .common import COLLT_SFT, SPECIAL_TOKENS, jsonl_iter


def _records(path: str | Path, limit: Optional[int] = None):
    for i, r in enumerate(jsonl_iter(path)):
        if limit is not None and i >= limit:
            break
        yield r


def build_chat_dataset(path: str | Path = COLLT_SFT,
                       limit: Optional[int] = None) -> Dataset:
    """Returns Dataset with a single `messages` column suitable for
    unsloth's `to_sharegpt`/`apply_chat_template` flow."""
    rows = []
    for rec in _records(path, limit):
        rows.append({
            "id":       rec["id"],
            "messages": rec["messages"],
            "num_clarification_turns": rec.get("num_clarification_turns", 0),
            "tools_used": rec.get("tools_used", []),
        })
    return Dataset.from_list(rows)


def build_pretokenized(path: str | Path,
                       tokenizer,
                       max_len: int = 2048,
                       limit: Optional[int] = None) -> Dataset:
    """Manual tokenization with assistant-only loss masking.

    The mask is built by walking each turn: user tokens get label = -100,
    assistant tokens keep their token-id as label, so cross-entropy is
    computed only on assistant spans (action+tool+response). This matches
    Equation 12 directly.
    """
    add_special_tokens_to(tokenizer)

    samples = []
    for rec in _records(path, limit):
        ids:    list[int] = []
        labels: list[int] = []
        for msg in rec["messages"]:
            chunk = tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False
            )
            ids.extend(chunk)
            if msg["role"] == "assistant":
                labels.extend(chunk)        # supervised
            else:
                labels.extend([-100] * len(chunk))  # masked

        ids    = ids[:max_len]
        labels = labels[:max_len]
        samples.append({"input_ids": ids, "labels": labels,
                        "attention_mask": [1] * len(ids)})
    return Dataset.from_list(samples)


def add_special_tokens_to(tokenizer) -> int:
    """Register the COLLT control tokens; returns number actually added."""
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    return added
