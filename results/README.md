# Results: Experiments 37–43 (class master dataset)

Everything behind the numbers in the docs, copied from the DGX by
`scripts/collect_results.sh` (`logs/` and `checkpoints/` stay out of git).

| Folder / file | What |
|---|---|
| `summary.md` | Mean ± standard deviation over 3 seeds, val / test / holdout, every config (`scripts/summarize_experiments.py`) |
| `train_logs/exp<config>_s<seed>.log` | Full training log per run: settings, per-epoch losses and accuracies, saved checkpoints |
| `eval/exp<config>_eval_<split>.txt` | `scripts/evaluate_checkpoint.py` output for the 3 seeds of a config on val, test or holdout |
| `eval/final43_onnx_eval_<split>.txt` | The shipped ONNX files (`models/vcm_intent.onnx`, `vcm_intent_small.onnx`) on test and holdout. `final_onnx_eval_*` are the Experiment 41d / 40b files on the first revision |
| `eval/*_on_r2_eval_test.txt`, `eval/exp43_shipped41d_on_r2_*.txt` | Models trained on the first revision, scored on the current test set |
| `eval/exp37_baseline_exp36onnx_*.txt`, `eval/exp37_eval_test_unseen_by_exp36.txt` | The previous (old-dataset) model on this test set, and on the test clips it never trained on |
| `launchers/` | The exact shell scripts that launched each experiment on GPU 6 of `ai-n002` |
| `bench_pi5.md` | The Raspberry Pi 5 measurements (latency p50/p95, RTF, memory) of the shipped models |
| `bench_dgx_1core_*.json` | `scripts/benchmark_pi.py` on one DGX CPU core (latency p50/p95, RTF), for comparison with the Pi |

The shipped wake word is Experiment 43 seed 1 (`train_logs/exp43w_wake_s1.log`, `eval/exp43w_eval_wake_*.txt`, checkpoint `models/kiwi_wakeword.pt`).
The shipped intent model is Experiment 43b seed 1 (`train_logs/exp43b_supplemental_s1.log`);
its checkpoint is `models/vcm_intent.pt`. What each config changes is in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md), Part 2.

Experiments 37–42 used the dataset's first revision (`25111444`); Experiment 43 the current one (`da92a79`). Their test numbers are not comparable.

`class_benchmark_offline/` is the offline rehearsal of the class's live benchmark (airimonda/vcm-benchmark at `eaf3605`), from `scripts/class_benchmark_offline.py`; see docs/TESTING.md.
