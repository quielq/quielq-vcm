# Training on the shared DGX — runbook

How training runs are launched on the shared DGX node (`ai-n002`,
8× A100-40GB, 256 cores, shared with 100+ other users), and why. Written
up from what was actually measured during Experiments 28–30
(EXPERIMENTS.md). The practices here aren't specific to this project;
they apply to any small-model training job on a shared multi-GPU node.

## TL;DR

1. **Pick one idle GPU.** Don't spread across GPUs other people are using.
2. **Pack several runs onto it** (3–5 for a model this size) instead of one run per GPU.
3. **Pin every process to one thread.** Otherwise DataLoader workers oversubscribe the node.
4. **Launch with `nohup` from a script,** with one log per run, so runs survive disconnects.
5. **Always run multiple seeds** (3), and compare on the held-out test metric, not best val.

## 1. Pick a GPU

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Pick a GPU with **no other processes on it** (a few hundred MiB used, 0%
utilization). A GPU at 0% utilization that still holds another user's
memory is *not* free: they can start work at any moment, and our jobs
would slow theirs down or cause an out-of-memory error. During
Experiments 28–30, GPU 2 was the only completely empty one, so everything
ran there.

## 2. Pack several runs onto one GPU

A small model (this project's CRNN is 96K params, ~2.8 GB of GPU memory
per run) cannot keep an A100 busy. A batch takes the GPU a few
milliseconds; the real bottleneck is the **CPU side**: reading audio,
trimming, augmenting, and computing log-mels in the DataLoader workers.
One run alone leaves the GPU mostly idle, so several runs share it
almost for free.

Measured on GPU 2:

| What was running on GPU 2 | Per-epoch time | GPU utilization |
|---|---:|---|
| 5 CRNN runs, no augmentation (Exp 28) | ~34 s | not saturated |
| 5 CRNN runs, with waveform augmentation (Exp 30) | ~63–66 s | CPU-bound on augmentation |
| **11 runs at once** (Exps 28 + 29 overlapping) | ~60–130 s | **99%, GPU-bound, runs slow each other down** |

Rules of thumb:
- **3–5 runs per GPU** for models under ~1M params. Watch
  `utilization.gpu`: once it sits near 100%, adding runs only slows
  every run down.
- If you need more runs than that, split across a *second genuinely
  idle* GPU rather than overloading one (Exps 28 + 29 overlapping on
  one GPU was the one real inefficiency here).
- Memory is rarely the limit at this size (5 × 2.8 GB of 40 GB).

## 3. Pin threads to 1 (important on a shared node)

librosa, NumPy/BLAS and numba each start **one thread per core by
default, in every DataLoader worker process**. On a 256-core node, 5
runs × 12 workers created **~24,000 threads using ~200 cores**, which
starves the other 130+ users. Setting one thread per process fixed it:
**~1,350 threads, ~40 cores, and epochs got about 2× faster**
(oversubscription was slowing our own runs too).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

Budget CPU as roughly `runs × --num-workers` cores. 8 workers per run
was enough to keep up with augmentation. Check your share with:

```bash
ps -eo user=,pcpu= | awk '$1 ~ /quiel/ {c+=$2} END {print c "% CPU"}'   # 100% = one core
uptime                                                                 # load average vs. 256 cores
```

## 4. Launch from a script, with `nohup`

Put the launch commands in a script file on the DGX and run it with
`bash`. Each run gets `nohup`, its own log, and its own checkpoint path,
so runs keep going after SSH disconnects, VPN drops, or your laptop
sleeping. Template (the actual Experiment 29 launcher):

```bash
#!/bin/bash
# Exp 29a/29b, 3 seeds each, all on GPU 2.
cd ~/quielq-vcm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2
COMMON="--model crnn --epochs 80 --batch-size 128 --lr 1e-3 --warmup-epochs 5 \
  --confusable-alpha 2.0 --num-workers 8 --window-s 5.0 --trim-silence"
for s in 0 1 2; do
  nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON \
    --out checkpoints/exp29a_crnn_trim5s_s$s.pt > logs/exp29a_crnn_trim5s_s$s.log 2>&1 &
  nohup .venv/bin/python -m vcm.train.train --seed $s $COMMON --wave-augment \
    --out checkpoints/exp29b_crnn_trim5s_waveaug_s$s.pt > logs/exp29b_crnn_trim5s_waveaug_s$s.log 2>&1 &
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
for f in logs/exp29*.log; do echo "$f: $(grep '^epoch' $f | tail -1)"; done
```

## 5. Multiple seeds, and the right metric

- **3 seeds per configuration.** Several earlier "wins" in EXPERIMENTS.md
  (e.g. 14 vs 15, and 27) were within run-to-run noise. With 3 seeds,
  a gain that holds across all of them (e.g. 29b's +4.6pp, spread <0.6pp)
  is real.
- **Select on val, report on test.** `train.py` keeps the best-val
  checkpoint; compare configurations with
  `scripts/evaluate_checkpoint.py` on the **test** split, **real speech
  only**. Val includes synthetic clips, which score 94–99% and hide
  real-speech weaknesses.

## 6. Where things live

| What | Where | How it moves |
|---|---|---|
| Code | Git (branch → PR) | Commit on the Mac, push, then `git fetch && git checkout <branch>` on the DGX. Avoid copying files with `rsync`: it leaves the DGX checkout showing uncommitted changes. |
| Checkpoints, logs | `checkpoints/`, `logs/` on the DGX, both **gitignored** | `scp` the ones you need to the Mac, e.g. to test live with `scripts/demo_infer.py` |
| Data | `data/` on the DGX only (~4.5 GB) | Not in git |

**Where to run Claude Code for long jobs**: a session running *on the
DGX* (opened over SSH in the desktop app) keeps working when the laptop
sleeps or the VPN drops. A session running on the Mac that reaches the
DGX with per-command `ssh` pauses whenever the Mac loses the VPN. The
training runs themselves survive either way, since they're under
`nohup`.
