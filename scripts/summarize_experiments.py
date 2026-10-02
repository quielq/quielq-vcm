#!/usr/bin/env python
"""Mean and spread over seeds for each experiment config, from the outputs
of scripts/evaluate_checkpoint.py saved as logs/exp<TAG>_eval_<split>.txt
(one file per config and split, all seeds in it).

Prints one markdown table row per config: overall and real-speech intent
accuracy on val (used to compare configs), test (the class's fixed test
set) and holdout (the Raspberry Pi live-test set).

Usage:
    python scripts/summarize_experiments.py 37a_me2_e80 39a_freqmask 39b_speedwide ...
    python scripts/summarize_experiments.py --glob '39*'
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

SPLITS = ("val", "test", "holdout")
METRICS = (("all", r"^all:\s+([\d.]+)%"), ("real", r"^real speech:\s+([\d.]+)%"))


def parse(path: Path) -> dict[str, list[float]]:
    """{metric: [one value per checkpoint]} from one evaluate_checkpoint.py output."""
    out: dict[str, list[float]] = {name: [] for name, _ in METRICS}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        for name, pattern in METRICS:
            m = re.match(pattern, line)
            if m:
                out[name].append(float(m.group(1)))
    return out


def fmt(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--glob", default=None, help="Config tags matching logs/exp<GLOB>_eval_val.txt")
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    args = parser.parse_args()

    tags = list(args.tags)
    if args.glob:
        tags += sorted(p.name[3 : -len("_eval_val.txt")] for p in args.logs.glob(f"exp{args.glob}_eval_val.txt"))
    header = ["config"] + [f"{split} {name}" for split in SPLITS for name, _ in METRICS]
    print("| " + " | ".join(header) + " |")
    print("|---" + "|---:" * (len(header) - 1) + "|")
    for tag in tags:
        cells = [tag]
        for split in SPLITS:
            result = parse(args.logs / f"exp{tag}_eval_{split}.txt")
            cells += [fmt(result[name]) for name, _ in METRICS]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
