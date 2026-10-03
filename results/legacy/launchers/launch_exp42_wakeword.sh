#!/bin/bash
# Exp 42: "Hey Kiwi" wake word retrained with the class master dataset as its negatives
# (train split commands + out-of-scope speech + noise, 3,000 numerals clips); positives
# and near-misses unchanged (synthetic wakeword batch + the author's recordings).
# Exp 34 recipe otherwise, 3 seeds, GPU 6. Then streaming evaluation on the master test split.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
for s in 0 1 2; do
  .venv/bin/python scripts/train_wakeword.py --seed $s --intent-manifest data/me2/manifest.csv \
    --numerals-clips 3000 --extra-wake-manifest data/wakeword_real/manifest.csv \
    --out checkpoints/exp42_wake_me2_s$s.pt > logs/exp42_wake_me2_s$s.log 2>&1 &
done
# Baseline: the shipped Exp 34 wake word on the same evaluation
.venv/bin/python scripts/evaluate_wakeword.py models/kiwi_wakeword.onnx --intent-manifest data/me2/manifest.csv \
  --extra-wake-manifest data/wakeword_real/manifest.csv > logs/exp42_eval_wake_exp34.txt 2>&1
wait
for s in 0 1 2; do
  .venv/bin/python scripts/evaluate_wakeword.py checkpoints/exp42_wake_me2_s$s.pt --intent-manifest data/me2/manifest.csv \
    --extra-wake-manifest data/wakeword_real/manifest.csv > logs/exp42_eval_wake_me2_s$s.txt 2>&1
done
echo "exp42 done $(date +%T)"
