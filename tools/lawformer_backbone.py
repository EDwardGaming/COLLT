"""
Shared Lawformer loading + sequence-classification head wrapper.

Lawformer is a Longformer-style Chinese legal encoder (HuggingFace repo
`thunlp/Lawformer`). All three retrieval / classification tools in §4.1 of
the manuscript share this backbone:

  - T_SCR  (Siamese contrastive + triplet, see train_tscr.py)
  - T_LAS  (multi-class softmax over legal articles)
  - T_LCP  (multi-label sigmoid over charges)
"""
from __future__ import annotations
import os

from ..common import local_model_path

LAWFORMER_ID = os.environ.get("LAWFORMER_ID", "thunlp/Lawformer")


def _resolve():
    """本地优先,失败回退到 HF。"""
    return local_model_path(LAWFORMER_ID)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_resolve(), trust_remote_code=True)


def load_for_classification(num_labels: int, problem_type: str = "single_label_classification"):
    """problem_type ∈ {single_label_classification, multi_label_classification}"""
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        _resolve(),
        num_labels   = num_labels,
        problem_type = problem_type,
        trust_remote_code = True,
        ignore_mismatched_sizes = True,
    )


def load_for_encoding():
    from transformers import AutoModel
    return AutoModel.from_pretrained(_resolve(), trust_remote_code=True)
