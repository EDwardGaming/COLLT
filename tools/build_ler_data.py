"""
T_LER 训练数据下载与构建。

CAIL2019 元素识别任务(Element Extraction)的官方仓库
(china-ai-law-challenge/CAIL2019)只发布了标签定义文件 tags.txt /
selectedtags.txt,而真实的句子-标签数据未直接放在仓库里。第三名方案
HuiResearch/cail2019_track2 公开了**全部三个领域**的训练数据(divorce
/ labor / loan),格式与官方 baseline (svm.py) 期望的一致:

    每个 .json 文件 = JSONL,每行 = 一个判决书的 JSON 数组,
    每个元素 = {"labels": ["DV1", "DV3"], "sentence": "..."}.

本脚本做三件事:

  1. 下载三个领域的训练 / 验证文件到 local_data/CAIL2019/要素识别/。
     (若已存在则跳过)。
  2. 把上述嵌套结构平铺成「逐句多标签」的 JSONL
     ({"sentence":..., "labels":[...], "domain":...}),
     train_selected.json   → outputs/tler_train.jsonl
     data_small_selected.json → outputs/tler_valid.jsonl
  3. 读取 selectedtags.txt,生成 label→中文名映射,落到
     outputs/tler_label_names.json,供推理时把 "DV3" 解码为
     「有夫妻共同财产」。

论文里只用 divorce 子集(婚姻家庭),默认也只构建 divorce;通过
--domains divorce,labor,loan 可扩到全 3 域 (扩大数据量,提升泛化)。

用法:
    python -m train.tools.build_ler_data                       # 默认 divorce
    python -m train.tools.build_ler_data --domains divorce,labor,loan
    python -m train.tools.build_ler_data --force-download
"""
from __future__ import annotations
import argparse, json, urllib.request, urllib.error
from pathlib import Path

from ..common import OUT_DIR, TRAIN_DIR


# raw GitHub URLs for the 3rd-place CAIL2019-track2 solution (full dataset)
_BASE = "https://raw.githubusercontent.com/HuiResearch/cail2019_track2/master/data"
_FILES = ["train_selected.json", "data_small_selected.json",
          "tags.txt", "selectedtags.txt"]

# local mirror (under repo)
CAIL2019_ER_DIR = TRAIN_DIR.parent / "local_data" / "CAIL2019" / "要素识别"

TLER_TRAIN_OUT = OUT_DIR / "tler_train.jsonl"
TLER_VALID_OUT = OUT_DIR / "tler_valid.jsonl"
TLER_NAMES_OUT = OUT_DIR / "tler_label_names.json"


def _download(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"  [skip] {dst.relative_to(TRAIN_DIR.parent)} (exists)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        dst.write_bytes(data)
        print(f"        → {dst.relative_to(TRAIN_DIR.parent)} ({len(data)/1024:.0f} KB)")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"[T_LER] 下载失败: {url}\n"
            f"        原因: {e}\n"
            f"        请手动下载到 {dst}, 或检查网络后重试。"
        )


def _ensure_raw(domains: list[str], force: bool) -> None:
    """把 train_selected.json / data_small_selected.json / *tags.txt 拉到本地。"""
    print(f"[T_LER] 下载 CAIL2019 元素识别原始数据 (domains={domains})")
    for dom in domains:
        for fname in _FILES:
            url = f"{_BASE}/{dom}/{fname}"
            dst = CAIL2019_ER_DIR / dom / fname
            _download(url, dst, force)


def _iter_sentences(path: Path, domain: str):
    """CAIL2019 element 文件:每行是 list[{labels, sentence}]。平铺产出。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(doc, list):
                continue
            for sent in doc:
                text = sent.get("sentence", "").strip()
                labels = sent.get("labels", []) or []
                if not text:
                    continue
                yield {"sentence": text, "labels": labels, "domain": domain}


def _flatten(domains: list[str]) -> None:
    """把每个 domain 的 train/valid 文件平铺到 outputs/tler_*.jsonl。"""
    n_tr = n_va = 0
    label_set: set[str] = set()
    with open(TLER_TRAIN_OUT, "w", encoding="utf-8") as ftr, \
         open(TLER_VALID_OUT, "w", encoding="utf-8") as fva:
        for dom in domains:
            tr_src = CAIL2019_ER_DIR / dom / "train_selected.json"
            va_src = CAIL2019_ER_DIR / dom / "data_small_selected.json"
            for rec in _iter_sentences(tr_src, dom):
                ftr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                label_set.update(rec["labels"])
                n_tr += 1
            for rec in _iter_sentences(va_src, dom):
                fva.write(json.dumps(rec, ensure_ascii=False) + "\n")
                label_set.update(rec["labels"])
                n_va += 1
    print(f"[T_LER] 平铺完成: {n_tr} 条训练句, {n_va} 条验证句, "
          f"{len(label_set)} 种标签 → {OUT_DIR}/tler_*.jsonl")


def _emit_label_names(domains: list[str]) -> None:
    """读 tags.txt + selectedtags.txt,生成 {DV1: '婚后有子女', ...}。"""
    mapping: dict[str, str] = {}
    for dom in domains:
        tags = (CAIL2019_ER_DIR / dom / "tags.txt").read_text(encoding="utf-8").splitlines()
        names = (CAIL2019_ER_DIR / dom / "selectedtags.txt").read_text(encoding="utf-8").splitlines()
        tags = [t.strip() for t in tags if t.strip()]
        names = [n.strip() for n in names if n.strip()]
        for t, n in zip(tags, names):
            mapping[t] = n
    TLER_NAMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TLER_NAMES_OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[T_LER] 标签中文名映射 ({len(mapping)} 条) → {TLER_NAMES_OUT}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domains", default="divorce",
                   help="逗号分隔的领域子集,可选 divorce,labor,loan")
    p.add_argument("--force-download", action="store_true",
                   help="即使本地已有也重新下载")
    args = p.parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    assert all(d in {"divorce", "labor", "loan"} for d in domains), \
        f"不支持的 domain: {domains}"

    _ensure_raw(domains, args.force_download)
    _flatten(domains)
    _emit_label_names(domains)


if __name__ == "__main__":
    main()
