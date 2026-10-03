# Training on the shared DGX — runbook

How training runs are launched on the shared DGX node (`ai-n002`,
8× A100-40GB, 256 cores, shared with 100+ other users), and why, then the
exact commands that reproduce the final model. The practices come from
what we measured on the node; they aren't specific to this project and
apply to any small-model training job on a shared multi-GPU node.

## TL;DR

1. **Pick one idle GPU.** Don't spread across GPUs other people are using.
2. **Pack several runs onto it** (3–5 for a model this size) instead of one run per GPU.
3. **Pin every process to one thread.** Otherwise DataLoader workers oversubscribe the node.
4. **Launch with `nohup` from a script,** with one log per run, so runs survive disconnects.
5. **Always run multiple seeds** (3). Choose on val (real speech), report on test.

## 1. Pick a GPU

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Pick a GPU with **no other processes on it** (a few hundred MiB used, 0%
utilization). A GPU at 0% utilization that still holds another user's
memory is *not* free: they can start work at any moment, and our jobs
would slow theirs down or cause an out-of-memory error. The final model's
runs all went on GPU 6, the one idle GPU at the time.

## 2. Pack several runs onto one GPU

A small model (the final CRNN is 372K params, ~2.5 GB of GPU memory per
run) cannot keep an A100 busy. A batch takes the GPU a few milliseconds;
the real bottleneck is the **CPU side**: reading audio, trimming,
augmenting, and computing log-mels in the DataLoader workers. One run
alone leaves the GPU mostly idle, so several runs share it almost for
free. For the final model, 9 runs and another user's small job shared one
A100 at ~40 s per epoch each; with fewer runs sharing it, ~25 s. Packing
11 runs onto one GPU earlier in the project pushed it to 99% utilization,
and every run slowed down (the measurements are in
[EXPERIMENTS.md](EXPERIMENTS.md#training-runs-on-the-shared-dgx-earlier-experiments)).

Rules of thumb:
- **3–5 runs per GPU** for models under ~1M params. Watch
  `utilization.gpu`: once it sits near 100%, adding runs only slows
  every run down.
- If you need more runs than that, split across a *second genuinely
  idle* GPU rather than overloading one.
- Memory is rarely the limit at this size (9 × 2.5 GB of 40 GB).

## 3. Pin threads to 1 (important on a shared node)

librosa, NumPy/BLAS and numba each start **one thread per core by
default, in every DataLoader worker process**. On a 256-core node, 5
runs × 12 workers once created **~24,000 threads using ~200 cores**, which
starves the other 130+ users. Setting one thread per process fixed it:
**~1,350 threads, ~40 cores, and epochs got about 2× faster**
(oversubscription was slowing our own runs too).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

Budget CPU as roughly `runs × --num-workers` cores. 6 workers per run
kept up with augmentation for the final model. Check your share with:

```bash
ps -eo user=,pcpu= | awk '$1 ~ /quiel/ {c+=$2} END {print c "% CPU"}'   # 100% = one core
uptime                                                                 # load average vs. 256 cores
```

## 4. Launch from a script, with `nohup`

Put the launch commands in a script file on the DGX and run it with
`bash`. Each run gets `nohup`, its own log, and its own checkpoint path,
so runs keep going after SSH disconnects, VPN drops, or your laptop
sleeping. Template (a shortened version of the final model's launcher,
[`results/launchers/run_queue_exp43.sh`](../results/launchers/run_queue_exp43.sh)):

```bash
#!/bin/bash
# Final recipe (Exp 43b), 3 seeds, all on one idle GPU.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
M=data/me2/manifest.csv
COMMON="--model crnn --manifest $M --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6 \
  --rnn-layers 2 --pool-heads 4 --width 80 --rnn-hidden 96 --augment --freq-mask-only \
  --babble-manifest $M --babble-clips 0 --extra-oos-clips 1500 --include-supplemental \
  --distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher \
  --distill-labels data/me2/ensemble43_labels.csv"
for s in 0 1 2; do
  nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON \
    --out checkpoints/exp43b_supplemental_s$s.pt > logs/exp43b_supplemental_s$s.log 2>&1 &
done
```

Two traps:
- **Don't `pkill -f <pattern>` inside an `ssh '...'` one-liner** that
  also contains the pattern. The pattern matches the SSH shell's own
  command line, so it kills itself before the relaunch half runs. Use a
  script file, or `pgrep -f "python -m vcm.train"` to list the PIDs and
  kill those.
- **Smoke-test first**: `--epochs 1 --train-fraction 0.02` catches
  crashes in about a minute instead of after a full launch.

Monitor progress without attaching to anything:

```bash
for f in logs/exp43*.log; do echo "$f: $(grep '^epoch' $f | tail -1)"; done
```

## 5. Multiple seeds, and the right metric

- **3 seeds per configuration.** Seeds of the final recipe differ by up to
  2 points on real speech (test real 76.71 / 78.64 / 77.35%), so a gain
  smaller than that from one seed is noise. Compare means over seeds.
- **Select on val, report on test.** `train.py` keeps the best-val
  checkpoint; compare configurations with
  `scripts/evaluate_checkpoint.py` on the **val** split, looking at
  **real speech** as well as all clips: synthetic clips score ~99% and
  hide real-speech weaknesses. Test and holdout are only reported, never
  used to choose.

## 6. Where things live

| What | Where | How it moves |
|---|---|---|
| Code | Git (branch → PR) | Commit on the Mac, push, then `git fetch && git checkout <branch>` on the DGX. Avoid copying files with `rsync`: it leaves the DGX checkout showing uncommitted changes. |
| Checkpoints, logs | `checkpoints/`, `logs/` on the DGX, both **gitignored** | `scp` the ones you need to the Mac, e.g. to export with `scripts/export_onnx.py` and test live with `scripts/vcm_listen.py --intent-model` |
| Data | The class's shared copy `/data/ai231` on the DGX (or a ~3.6 GB pinned download), built into `data/me2/` | Not in git ([DATASET.md](DATASET.md)) |

**Where to run Claude Code for long jobs**: a session running *on the
DGX* (opened over SSH in the desktop app) keeps working when the laptop
sleeps or the VPN drops. A session running on the Mac that reaches the
DGX with per-command `ssh` pauses whenever the Mac loses the VPN. The
training runs themselves survive either way, since they're under
`nohup`.

## Reproducing the final model

The final model trains on the class master dataset
([DATASET.md](DATASET.md)). The exact launcher is
[`results/launchers/run_queue_exp43.sh`](../results/launchers/run_queue_exp43.sh).
All runs went on one idle GPU (GPU 6 at the time).

```bash
# Data (once): download the master dataset and build data/me2/
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  'airimonda/ai231-me2-voice-commands', repo_type='dataset', local_dir='data/me2/hf',
  revision='da92a79ffde3031d5bb2a25138d9dd7d9f7ed006', allow_patterns=['data/*', 'supplemental_synth/*', 'README.md', 'variations.csv'])"
python scripts/build_me2_manifest.py --numerals --supplemental
# On the DGX, use the class's shared copy instead, no download (identical to da92a79
# plus its supplemental_synth; DATASET.md):
#   python scripts/verify_shared_dataset.py --shared-cache /data/ai231
#   python scripts/build_me2_manifest.py --shared-cache /data/ai231 --numerals --supplemental

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=<an idle GPU>

# Distillation teacher (Experiment 43t): 9 smaller CRNNs on the same train split, the
# base flags only (no --rnn-layers/--pool-heads/--width/...): 80 epochs, 150 epochs, and
# 80 epochs --real-oversample 3, seeds 0-2 each (see the launcher); then their soft labels
python scripts/generate_ensemble_labels.py checkpoints/exp43t_{e80,e150,real3}_s{0,1,2}.pt \
  --manifest data/me2/manifest.csv --out data/me2/ensemble43_labels.csv

# Intent + slot model (Experiment 43b; seed 1 shipped, seeds 0 and 2 for comparison)
python -m vcm.train.train --model crnn --seed 1 \
  --manifest data/me2/manifest.csv --slot-labels data/me2/slot_labels.csv --slot-weight 0.3 \
  --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 --confusable-alpha 2.0 \
  --window-s 5.0 --trim-silence --wave-augment --num-workers 6 \
  --rnn-layers 2 --pool-heads 4 --width 80 --rnn-hidden 96 --augment --freq-mask-only \
  --babble-manifest data/me2/manifest.csv --babble-clips 0 --extra-oos-clips 1500 \
  --include-supplemental \
  --distill-weight 1.0 --distill-temperature 3 --distill-soften-teacher \
  --distill-labels data/me2/ensemble43_labels.csv \
  --out checkpoints/exp43b_supplemental_s1.pt

# Evaluate on the class-fixed test set and the Pi holdout set, then export
python scripts/evaluate_checkpoint.py checkpoints/exp43b_supplemental_s*.pt \
  --manifest data/me2/manifest.csv --slot-labels data/me2/slot_labels.csv \
  --metadata data/me2/metadata.csv --split test        # and --split holdout
python scripts/export_onnx.py checkpoints/exp43b_supplemental_s1.pt --out models/vcm_intent
# The small model (Experiment 43c seed 0): the same command without --width 80 --rnn-hidden 96
# and --include-supplemental, exported to models/vcm_intent_small

# "Hey Kiwi" wake word (Experiment 43, seed 1 shipped): master dataset speech and noise
# as negatives; positives from the synthetic wakeword batch (data/external/wakeword_synth,
# published in the quielq-vcm-dataset v1.0 release) and data/wakeword_real
python scripts/train_wakeword.py --seed 1 --intent-manifest data/me2/manifest.csv \
  --numerals-clips 3000 --extra-wake-manifest data/wakeword_real/manifest.csv \
  --out checkpoints/exp43w_wake_s1.pt
python scripts/evaluate_wakeword.py checkpoints/exp43w_wake_s1.pt --intent-manifest data/me2/manifest.csv \
  --extra-wake-manifest data/wakeword_real/manifest.csv
python scripts/export_onnx.py checkpoints/exp43w_wake_s1.pt --out models/kiwi_wakeword
```

`bash scripts/reproduce.sh` does all of the above in one command.

Timing on `ai-n002`, everything on one A100 (GPU 6), with 9 runs and
another user's job sharing the GPU: ~40 s per epoch, ~54 min per seed of
the final recipe; the whole retrain (9 teachers, 3 wake words, 9 final runs
and their evaluations) took about 2.5 hours. Results vary by 0.1–2 points
between seeds (real speech varies most), so every configuration ran 3
seeds and was compared on val. The CRNN needs ~2.5 GB of GPU memory per
run. All logs, evaluations and launchers are in [`results/`](../results/).
