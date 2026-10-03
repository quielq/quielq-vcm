#!/bin/bash
# Exp 37: CRNN + slot heads (the Exp 36 recipe) retrained on the class master
# dataset (data/me2, scripts/build_me2_manifest.py). 37a = 80 epochs as shipped,
# 37b = 150 epochs (the new train split is ~7x smaller). 3 seeds each, all on GPU 6.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
COMMON="--model crnn --manifest data/me2/manifest.csv --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
for s in 0 1 2; do
  nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON --epochs 80 \
    --out checkpoints/exp37a_me2_e80_s$s.pt > logs/exp37a_me2_e80_s$s.log 2>&1 &
  nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON --epochs 150 \
    --out checkpoints/exp37b_me2_e150_s$s.pt > logs/exp37b_me2_e150_s$s.log 2>&1 &
done

# 37c (launched after 37a finished): 37a + --real-oversample 3 (real clips are
# 27% of train). Same GPU.
# for s in 0 1 2; do nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON --epochs 80 \
#   --real-oversample 3 --out checkpoints/exp37c_me2_real3_s$s.pt > logs/exp37c_me2_real3_s$s.log 2>&1 & done
