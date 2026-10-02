# Results: Experiments 37–42 (class master dataset)

Everything behind the numbers in the docs, copied from the DGX by
`scripts/collect_results.sh` (`logs/` and `checkpoints/` stay out of git).

| Folder / file | What |
|---|---|
| `summary.md` | Mean ± standard deviation over 3 seeds, val / test / holdout, every config (`scripts/summarize_experiments.py`) |
| `train_logs/exp<config>_s<seed>.log` | Full training log per run: settings, per-epoch losses and accuracies, saved checkpoints |
| `eval/exp<config>_eval_<split>.txt` | `scripts/evaluate_checkpoint.py` output for the 3 seeds of a config on val, test or holdout |
| `eval/final_onnx_eval_<split>.txt` | The shipped ONNX files (`models/vcm_intent.onnx`, `vcm_intent_small.onnx`) on test and holdout |
| `eval/exp37_baseline_exp36onnx_*.txt`, `eval/exp37_eval_test_unseen_by_exp36.txt` | The previous (old-dataset) model on this test set, and on the test clips it never trained on |
| `launchers/` | The exact shell scripts that launched each experiment on GPU 6 of `ai-n002` |
| `bench_dgx_1core_*.json` | `scripts/benchmark_pi.py` on one DGX CPU core (latency p50/p95, RTF), for comparison with the Pi |

The shipped wake word is Experiment 42 seed 1 (`train_logs/exp42_wake_me2_s1.log`, `eval/exp42_eval_wake_*.txt`, checkpoint `models/kiwi_wakeword.pt`).
The shipped intent model is Experiment 41d seed 0 (`train_logs/exp41d_combo_wide_distill_s0.log`);
its checkpoint is `models/vcm_intent.pt`. What each config changes is in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md), Part 2.
