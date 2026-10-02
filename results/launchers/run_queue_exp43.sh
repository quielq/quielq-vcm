#!/bin/bash
# Exp 43: retrain on the master dataset's 2026-10-02 revision (da92a79), all on GPU 6.
#   A: 9 distillation teachers (Exp 37a/b/c recipes x seeds 0-2) + wake word (Exp 42 recipe, 3 seeds)
#   B: ensemble labels, then 3 seeds each of
#        43a = the shipped recipe (Exp 41d)
#        43b = 43a + the dataset's supplemental synthetic clips of train voices
#        43c = the small model's recipe (Exp 40b)
#   C: evaluation on val / test / holdout (CPU)
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
M=data/me2/manifest.csv
BASE="--model crnn --manifest $M --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6"
COMBO="--rnn-layers 2 --pool-heads 4 --augment --freq-mask-only --babble-manifest $M --babble-clips 0 --extra-oos-clips 1500"
WIDE="--width 80 --rnn-hidden 96"
D="--distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher --distill-labels data/me2/ensemble43_labels.csv"

evaluate() {  # $1 = config tag, $2 = extra flags
  for split in val test holdout; do
    OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_checkpoint.py checkpoints/exp$1_s{0,1,2}.pt --manifest $M \
      $2 --metadata data/me2/metadata.csv --split $split --num-workers 8 --device cpu > logs/exp$1_eval_$split.txt 2>&1
  done
}

# A
for s in 0 1 2; do
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 80 --out checkpoints/exp43t_e80_s$s.pt > logs/exp43t_e80_s$s.log 2>&1 &
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 150 --out checkpoints/exp43t_e150_s$s.pt > logs/exp43t_e150_s$s.log 2>&1 &
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 80 --real-oversample 3 --out checkpoints/exp43t_real3_s$s.pt > logs/exp43t_real3_s$s.log 2>&1 &
  .venv/bin/python scripts/train_wakeword.py --seed $s --intent-manifest $M --numerals-clips 3000 \
    --extra-wake-manifest data/wakeword_real/manifest.csv --out checkpoints/exp43w_wake_s$s.pt > logs/exp43w_wake_s$s.log 2>&1 &
done
wait
echo "phase A done $(date +%T)"
for s in 0 1 2; do
  .venv/bin/python scripts/evaluate_wakeword.py checkpoints/exp43w_wake_s$s.pt --intent-manifest $M \
    --extra-wake-manifest data/wakeword_real/manifest.csv > logs/exp43w_eval_wake_s$s.txt 2>&1 &
done
.venv/bin/python scripts/generate_ensemble_labels.py checkpoints/exp43t_{e80,e150,real3}_s{0,1,2}.pt \
  --manifest $M --out data/me2/ensemble43_labels.csv > logs/exp43_ensemble.log 2>&1
for t in e80 e150 real3; do evaluate 43t_$t "--slot-labels data/me2/slot_labels.csv" & done

# B
for s in 0 1 2; do
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 80 $COMBO $WIDE $D \
    --out checkpoints/exp43a_shipped_recipe_s$s.pt > logs/exp43a_shipped_recipe_s$s.log 2>&1 &
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 80 $COMBO $WIDE $D --include-supplemental \
    --out checkpoints/exp43b_supplemental_s$s.pt > logs/exp43b_supplemental_s$s.log 2>&1 &
  .venv/bin/python -m vcm.train.train --seed $s $BASE --epochs 80 $COMBO $D \
    --out checkpoints/exp43c_small_recipe_s$s.pt > logs/exp43c_small_recipe_s$s.log 2>&1 &
done
wait
echo "phase B done $(date +%T)"

# C
for t in 43a_shipped_recipe 43b_supplemental 43c_small_recipe; do evaluate $t "--slot-labels data/me2/slot_labels.csv" & done
wait
echo "queue done $(date +%T)"
