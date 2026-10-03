#!/bin/bash
# Copy the final experiment's (Experiment 43) training logs, evaluation outputs and
# launcher from the DGX's logs/ into results/ (committed to git; logs/ itself is
# gitignored). Run from the repo root. Experiments 37-42 are already in
# results/legacy/ and don't change.
set -e
mkdir -p results/train_logs results/eval results/launchers
cp logs/exp43*_s[0-9].log logs/exp43_ensemble.log results/train_logs/ 2>/dev/null || true
cp logs/exp43[a-z]*_eval_*.txt logs/exp43w_eval_wake_s[0-9].txt results/eval/ 2>/dev/null || true
cp logs/run_queue_exp43.sh results/launchers/ 2>/dev/null || true
python scripts/summarize_experiments.py --logs results/eval --glob '43*' > results/summary.md
du -sh results
