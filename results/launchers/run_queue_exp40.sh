#!/bin/bash
# Exp 40: combine the Exp 39 changes that didn't hurt on val, 3 seeds each, GPU 6.
#   40a = 2-layer GRU + 4 attention heads + frequency-only SpecAugment + 1,500 numerals clips as OUT_OF_SCOPE
#   40b = 40a + self-distillation from the 9-model Exp 37 ensemble (T=3, softened teacher, weight 1)
# Starts when the Exp 39 queue is done.
cd ~/quielq-vcm
until grep -q "queue done" logs/run_queue_exp39.out; do sleep 30; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
M=data/me2/manifest.csv
BASE="--model crnn --manifest $M --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
COMBO="--rnn-layers 2 --pool-heads 4 --augment --freq-mask-only --babble-manifest $M --babble-clips 0 --extra-oos-clips 1500"
CONFIGS=(
  "40a_combo|$COMBO"
  "40b_combo_distill|$COMBO --distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher --distill-labels data/me2/ensemble37_labels.csv"
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
for c in "${CONFIGS[@]}"; do run_config "${c%%|*}" "${c#*|}" & done
wait
echo "queue done $(date +%T)"
