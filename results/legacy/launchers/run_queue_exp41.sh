#!/bin/bash
# Exp 41: build on 40a (the combo), 3 seeds each, GPU 6, two configs (6 runs) at a time.
cd ~/quielq-vcm
until grep -q "queue done" logs/run_queue_exp40.out; do sleep 30; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
M=data/me2/manifest.csv
BASE="--model crnn --manifest $M --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
COMBO="--rnn-layers 2 --pool-heads 4 --augment --freq-mask-only --babble-manifest $M --babble-clips 0 --extra-oos-clips 1500"
#   41a = 40a + wider (80 channels, GRU 96)                      ~300K params
#   41b = 40a + distillation from the 6-model Exp 40 ensemble (T=3)
#   41c = 40b with 4,000 numerals clips as OUT_OF_SCOPE (was 1,500)
#   41d = 41a + distillation from the Exp 37 ensemble (as 40b)
D37="--distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher --distill-labels data/me2/ensemble37_labels.csv"
D40="--distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher --distill-labels data/me2/ensemble40_labels.csv"
CONFIGS=(
  "41a_combo_wide|$COMBO --width 80 --rnn-hidden 96"
  "41b_combo_distill40|$COMBO $D40"
  "41c_combo_distill_oos4k|${COMBO/--extra-oos-clips 1500/--extra-oos-clips 4000} $D37"
  "41d_combo_wide_distill|$COMBO --width 80 --rnn-hidden 96 $D37"
)
run_config() {
  local tag=$1 flags=$2
  for s in 0 1 2; do
    .venv/bin/python -m vcm.train.train --seed $s $BASE $flags \
      --out checkpoints/exp${tag}_s$s.pt > logs/exp${tag}_s$s.log 2>&1 &
  done
  wait
  for split in val test holdout; do
    OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_checkpoint.py checkpoints/exp${tag}_s{0,1,2}.pt --manifest $M \
      --slot-labels data/me2/slot_labels.csv --metadata data/me2/metadata.csv --split $split \
      --num-workers 8 --device cpu > logs/exp${tag}_eval_$split.txt 2>&1
  done
  echo "config $tag done $(date +%T)"
}
for ((i = 0; i < ${#CONFIGS[@]}; i += 2)); do
  for j in $i $((i + 1)); do
    run_config "${CONFIGS[$j]%%|*}" "${CONFIGS[$j]#*|}" &
  done
  wait
done
echo "queue done $(date +%T)"
