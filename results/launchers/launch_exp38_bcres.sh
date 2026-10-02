#!/bin/bash
# Exp 38 BC-ResNet baseline, rerun one seed at a time (~12 GB each at 112 channels; the
# first attempt ran out of memory next to the Exp 39 runs on GPU 6). Then evaluate on CPU.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
COMMON="--manifest data/me2/manifest.csv --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 \
  --confusable-alpha 2.0 --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
for s in 0 1 2; do
  .venv/bin/python -m vcm.train.train --seed $s --model bcresnet --width 112 --depth 6 $COMMON \
    --out checkpoints/exp38_bcres112x6_s$s.pt > logs/exp38_bcres112x6_s$s.log 2>&1
done
export OMP_NUM_THREADS=4
for split in val test holdout; do
  .venv/bin/python scripts/evaluate_checkpoint.py checkpoints/exp38_bcres112x6_s{0,1,2}.pt --manifest data/me2/manifest.csv \
    --metadata data/me2/metadata.csv --split $split --num-workers 8 --device cpu > logs/exp38_bcres112x6_eval_$split.txt 2>&1
done
echo "bcres done $(date +%T)"
