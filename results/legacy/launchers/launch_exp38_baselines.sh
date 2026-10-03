#!/bin/bash
# Exp 38: comparable-size baselines (~100K params) on the master dataset, intent only,
# same recipe as Exp 37a (80 epochs). DS-CNN 128x5 (99.6K), BC-ResNet 112ch x6 (89.4K).
# GPU 6. One family at a time: 3 runs of these full-resolution models use ~21 GB.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
COMMON="--manifest data/me2/manifest.csv --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 \
  --confusable-alpha 2.0 --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
for s in 0 1 2; do
  .venv/bin/python -m vcm.train.train --seed $s --model dscnn --width 128 --depth 5 $COMMON \
    --out checkpoints/exp38_dscnn128x5_s$s.pt > logs/exp38_dscnn128x5_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  .venv/bin/python -m vcm.train.train --seed $s --model bcresnet --width 112 --depth 6 $COMMON \
    --out checkpoints/exp38_bcres112x6_s$s.pt > logs/exp38_bcres112x6_s$s.log 2>&1 &
done
wait
