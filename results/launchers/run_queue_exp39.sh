#!/bin/bash
# Exp 39: one change at a time on top of Exp 37a (CRNN + slots, master dataset, 80 epochs),
# 3 seeds each, all on GPU 6, two configs (6 runs) at a time. Each config is evaluated on
# val (used to compare configs), test and holdout when its 3 seeds finish.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
M=data/me2/manifest.csv
BASE="--model crnn --manifest $M --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
CONFIGS=(
  "39a_freqmask|--augment --freq-mask-only"
  "39b_speedwide|--speed-range 0.85 1.15 --noise-prob 0.8"
  "39c_babble|--babble-manifest $M --babble-clips 2000"
  "39d_numoos|--babble-manifest $M --babble-clips 0 --extra-oos-clips 1500"
  "39e_ema|--ema-decay 0.999"
  "39f_heads4|--pool-heads 4"
  "39g_gru2|--rnn-layers 2"
  "39h_meanpool|--pool mean"
  "39i_ls01|--label-smoothing 0.1"
  "39j_wide|--width 80 --rnn-hidden 96"
)
run_config() {
  local tag=$1 flags=$2
  for s in 0 1 2; do
    .venv/bin/python -m vcm.train.train --seed $s $BASE $flags \
      --out checkpoints/exp${tag}_s$s.pt > logs/exp${tag}_s$s.log 2>&1 &
  done
  wait
  for split in val test holdout; do
    .venv/bin/python scripts/evaluate_checkpoint.py checkpoints/exp${tag}_s{0,1,2}.pt --manifest $M \
      --slot-labels data/me2/slot_labels.csv --metadata data/me2/metadata.csv --split $split \
      --num-workers 6 > logs/exp${tag}_eval_$split.txt 2>&1
  done
  echo "config $tag done $(date +%T)"
}
for ((i = 0; i < ${#CONFIGS[@]}; i += 2)); do
  for j in $i $((i + 1)); do
    [ $j -lt ${#CONFIGS[@]} ] || continue
    run_config "${CONFIGS[$j]%%|*}" "${CONFIGS[$j]#*|}" &
  done
  wait
done
echo "queue done $(date +%T)"
