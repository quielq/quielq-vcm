#!/bin/bash
# One command from a fresh clone to the shipped intent + slot model:
# download the class master dataset, build the manifests, train the final
# recipe, evaluate on the class test and holdout splits, export to ONNX and
# time it. Run from the repo root on a machine with a GPU. It first trains
# the 9 small teacher models for distillation (about 25 minutes each on an
# A100 if run alone; skipped if their checkpoints exist), then the final model.
#
#   bash scripts/reproduce.sh            # seed 1 only (the shipped seed)
#   SEEDS="0 1 2" bash scripts/reproduce.sh
#   GPU=3 bash scripts/reproduce.sh      # pick an idle GPU (default 0)
#
# Needs: python -m venv .venv && .venv/bin/pip install -e ".[dev,train,deploy]"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-1}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

# 1. Data: train/test/holdout, numerals and supplemental_synth (about 3.6 GB), then this repo's manifests.
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("airimonda/ai231-me2-voice-commands", repo_type="dataset", local_dir="data/me2/hf",
                  revision="da92a79ffde3031d5bb2a25138d9dd7d9f7ed006",  # the 2026-10-02 revision every result is on
                  allow_patterns=["data/*", "supplemental_synth/*", "README.md", "variations.csv"])
EOF
$PY scripts/build_me2_manifest.py --numerals --supplemental
M=data/me2/manifest.csv

# 2. Train (the final recipe, Experiment 43b; see docs/TRAINING.md and EXPERIMENTS.md).
# 2a. The distillation teacher: 9 CRNNs with the Experiment 36 recipe (80 epochs, 150 epochs,
#     80 epochs with real clips x3; seeds 0-2), averaged into soft labels on the train split.
TEACHERS=()
for cfg in "e80:--epochs 80" "e150:--epochs 150" "real3:--epochs 80 --real-oversample 3"; do
  tag=${cfg%%:*}; flags=${cfg#*:}
  for s in 0 1 2; do
    ckpt="checkpoints/teacher_${tag}_s$s.pt"
    [ -f "$ckpt" ] || $PY -m vcm.train.train --seed "$s" --model crnn --manifest $M \
      --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 --batch-size 128 --lr 1e-3 \
      --warmup-epochs 5 --confusable-alpha 2.0 --window-s 5.0 --trim-silence --wave-augment \
      --num-workers 6 $flags --out "$ckpt" > "logs/teacher_${tag}_s$s.log" 2>&1
    TEACHERS+=("$ckpt")
  done
done
$PY scripts/generate_ensemble_labels.py "${TEACHERS[@]}" --manifest $M --out data/me2/ensemble43_labels.csv
for s in $SEEDS; do
  $PY -m vcm.train.train --seed "$s" --model crnn --manifest $M \
    --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
    --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
    --window-s 5.0 --trim-silence --wave-augment --num-workers 6 \
    --rnn-layers 2 --pool-heads 4 --width 80 --rnn-hidden 96 --augment --freq-mask-only \
    --babble-manifest $M --babble-clips 0 --extra-oos-clips 1500 \
    --distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher \
    --include-supplemental \
    --distill-labels data/me2/ensemble43_labels.csv \
    --out "checkpoints/final_s$s.pt" 2>&1 | tee "logs/final_s$s.log"
done

# 3. Evaluate on the class-fixed test split and the Raspberry Pi holdout split.
for split in test holdout; do
  $PY scripts/evaluate_checkpoint.py $(for s in $SEEDS; do echo "checkpoints/final_s$s.pt"; done) \
    --manifest $M --slot-labels data/me2/slot_labels.csv --metadata data/me2/metadata.csv \
    --split $split | tee "logs/final_eval_$split.txt"
done

# 4. Export the first seed to ONNX and time it on this machine's CPU (run
#    scripts/benchmark_pi.py on the Pi itself for the Pi numbers).
first=$(echo $SEEDS | cut -d' ' -f1)
$PY scripts/export_onnx.py "checkpoints/final_s$first.pt" --out models/vcm_intent_reproduced
$PY scripts/benchmark_pi.py --intent-model models/vcm_intent_reproduced.onnx --clips data/me2/holdout/audio
