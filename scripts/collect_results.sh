#!/bin/bash
# Copy the training logs, evaluation outputs and launchers of Experiments 37 on into
# results/ (committed to git; logs/ itself is gitignored). Run from the repo root.
set -e
mkdir -p results/train_logs results/eval results/launchers
cp logs/exp3[789]*_s[0-9].log logs/exp4*_s[0-9].log results/train_logs/ 2>/dev/null || true
cp logs/exp3[789]*_eval_*.txt logs/exp4*_eval_*.txt logs/exp37_baseline_exp36onnx_*.txt results/eval/ 2>/dev/null || true
cp logs/launch_exp3[78]*.sh logs/run_queue_exp*.sh results/launchers/ 2>/dev/null || true
rm -f results/eval/exp37a_eval_*.txt results/eval/exp37bc_eval_*.txt  # duplicates of the per-config files
python scripts/summarize_experiments.py --logs results/eval --glob '3[789]*' > results/summary.md
python scripts/summarize_experiments.py --logs results/eval --glob '4*' >> results/summary.md
du -sh results
