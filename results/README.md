# Results: the final experiment (Experiment 43)

Everything behind the numbers in the docs, copied from the DGX by
`scripts/collect_results.sh` (`logs/` and `checkpoints/` stay out of git).
All of it is on the class master dataset, revision `da92a79`.

**The final model is Experiment 43b seed 1** (`train_logs/exp43b_supplemental_s1.log`,
checkpoint `models/vcm_intent.pt`); the wake word is Experiment 43 seed 1
(`train_logs/exp43w_wake_s1.log`, `eval/exp43w_eval_wake_s1.txt`, checkpoint
`models/kiwi_wakeword.pt`).

| Folder / file | What |
|---|---|
| `summary.md` | Mean ± standard deviation over 3 seeds, val / test / holdout, for every Experiment 43 config (`scripts/summarize_experiments.py`) |
| `train_logs/exp43<config>_s<seed>.log` | Full training log per run: settings, per-epoch losses and accuracies, saved checkpoints. Configs: `43b_supplemental` (final), `43c_small_recipe` (the small model), `43a_shipped_recipe` (43b without the supplemental clips), `43t_*` (the 9 distillation teachers), `43w_wake` (wake word). `exp43_ensemble.log` builds the teachers' soft labels; `build_me2_r2.log` builds the manifest |
| `eval/exp43<config>_eval_<split>.txt` | `scripts/evaluate_checkpoint.py` output for the 3 seeds of a config on val, test or holdout |
| `eval/final43_onnx_eval_<split>.txt` | The shipped ONNX files (`models/vcm_intent.onnx`, `vcm_intent_small.onnx`) on test and holdout |
| `eval/exp43w_eval_wake_s<seed>.txt` | The wake word's misses and false wake-ups per threshold |
| `launchers/run_queue_exp43.sh` | The exact script that ran Experiment 43 on GPU 6 of `ai-n002` |
| `bench_pi5.md` | The Raspberry Pi 5 measurements (latency p50/p95, RTF, memory) of the shipped models |
| `bench_dgx_1core_*.json` | `scripts/benchmark_pi.py` on one DGX CPU core, for comparison with the Pi |
| `class_benchmark_live/` | The class's live benchmark run on the Pi (2026-10-03); see docs/TESTING.md |
| `class_benchmark_offline/` | The offline rehearsal of that benchmark (`scripts/class_benchmark_offline.py`) |
| `legacy/` | Experiments 37–42 (logs, evaluations, launchers, their `summary.md`), on the dataset's first revision, kept for the project history ([docs/legacy/EXPERIMENTS.md](../docs/legacy/EXPERIMENTS.md), Part 2). Their test numbers are not comparable with Experiment 43's |
