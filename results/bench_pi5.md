# Raspberry Pi 5 measurements (2026-10-02)

Raspberry Pi 5 Model B Rev 1.1, 8 GB, Debian 12 (bookworm), kernel 6.12.109+rpt-rpi-2712.
Repo at master `11fc1d5`, deployed with `scripts/deploy_pi.sh raspberrypi.local --services`;
`kiwi_doctor.py` READY with intent `exp41d_combo_wide_distill_s0.pt` and wake word
`exp42_wake_me2_s1.pt`. Benchmark: `scripts/benchmark_pi.py --json`, ONNX Runtime CPU,
1 intra-op / 1 inter-op thread, synthetic 2.5 s command, n = 50, both services running.

| Intent model | Size | p50 ms | p95 ms | RTF p95 | Features ms | Model ms | Wake word, share of 1 core | Wake hop p95 ms | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcm_intent.onnx` (Exp 41d), run 2 | 1,463 KB | 13.93 | 15.78 | 0.0063 | 3.75 | 9.98 | 1.85% | 1.90 | 99.6 |
| `vcm_intent.onnx`, run 1 | 1,463 KB | 14.25 | 14.97 | 0.0060 | 3.93 | 10.27 | 1.94% | — | 99.3 |
| `vcm_intent_small.onnx` (Exp 40b) | 722 KB | 11.07 | 12.06 | 0.0048 | 3.93 | 7.21 | 1.98% | 2.01 | 98.3 |

Memory with both services running: 604 MB used of 8,063 MB (7,458 MB available, no swap);
`vcm_listen.py` 95.3 MB RSS, `vcm.home.server` 47.0 MB RSS.

These were measured with the Experiment 41d and 40b weights. The shipped
Experiment 43b and 43c models have exactly the same architectures and
operations (only the trained weights differ), so the timing and memory
carry over; re-run `scripts/benchmark_pi.py` on the Pi to confirm.

For comparison, the previous intent model (Exp 36, 432 KB) took 9.9 ms per command
(3.7 ms features + 6.2 ms model) on the same Pi.
