"""
工具消融实验(R3#7)。

设计方案(与作者沟通后):工具池有 6 种,完整组合 C(6,2)=15、C(6,3)=20…
枚举量太大,Table 3 也放不下。改用以下 8 行结构,直接覆盖审稿人意见:

    none                所有工具开,作为消融基线
    -SCR                关闭相似案例检索        (C(6,1) 之一)
    -LAS                关闭法条检索            (C(6,1) 之一)
    -LCP                关闭罪名预测            (C(6,1) 之一)
    -LER                关闭法律要素识别        (C(6,1) 之一)
    -LED                关闭法律事件检测        (C(6,1) 之一)
    -NET                关闭互联网搜索          (C(6,1) 之一)
    ALL                 全部工具关闭,仅保留澄清能力

实现上每一行复用 eval_table3.py(`--ablate` 透传 `COLLT_ABLATE`),所以
工具的"短路"既发生在 serve_tools(返回空串)也发生在流式层(头/尾标签
之间的内容被剥空),保证消融彻底。

用法:
    python -m train.eval_ablation_tools --ckpt checkpoints/collt-qwen
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

from .common import OUT_DIR


# 8 行配置:none 是基线,LAS/SCR/LCP/LER/LED/NET 是 C(6,1) 单工具消融,ALL = 全关
DEFAULT_CONFIGS = [
    "none", "SCR", "LAS", "LCP", "LER", "LED", "NET", "ALL",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="COLLT-* SFT checkpoint")
    ap.add_argument("--tasks", default="LCP,LAP,DFI,ITI,LED,OS,CA",
                    help="参与消融对比的 Table 3 任务子集(逗号分隔)")
    ap.add_argument("--mode", default="collt",
                    choices=["collt", "baseline_tools"],
                    help="collt = COLLT SFT 消融; "
                         "baseline_tools = 裸基线+工具 system prompt 后消融(R3#2 公平对照)")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟测试用,截断每个任务到 N 条")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="逗号分隔的消融组;每组内多工具用分号分隔 (e.g. LAS;SCR)")
    ap.add_argument("--out", default=str(OUT_DIR / "ablation_tools.json"))
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    report: dict[str, dict] = {}

    for cfg in configs:
        env = os.environ.copy()
        env["COLLT_ABLATE"] = "" if cfg == "none" else cfg.replace(";", ",")
        out_path = Path(args.out).with_suffix(f".{cfg.replace(',','_')}.json")
        cmd = [sys.executable, "-m", "train.eval_table3",
               "--ckpt", args.ckpt,
               "--mode", args.mode,
               "--tasks", args.tasks,
               "--out", str(out_path)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if cfg != "none":
            cmd += ["--ablate", env["COLLT_ABLATE"]]
        print(f"[ablate={cfg}] {' '.join(cmd)}")
        # cwd 要回到仓库根,以便 `python -m train.eval_table3` 找到包
        subprocess.run(cmd, check=True, env=env,
                       cwd=Path(__file__).resolve().parent.parent)
        report[cfg] = json.loads(out_path.read_text(encoding="utf-8"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
