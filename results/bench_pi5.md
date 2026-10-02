# Raspberry Pi 5 measurements (2026-10-02)

Raspberry Pi 5 Model B Rev 1.1, 8 GB, Debian 12 (bookworm), kernel 6.12.109+rpt-rpi-2712.
Benchmark: `scripts/benchmark_pi.py --json`, ONNX Runtime CPU, 1 intra-op / 1 inter-op
thread, synthetic 2.5 s command, n = 50, both services running, run twice and the second
run reported.

## Shipped models (Experiment 43b intent, Experiment 43 wake word)

Repo at master `d5d57b6`, deployed with `scripts/deploy_pi.sh raspberrypi.local --services`;
`kiwi_doctor.py` READY with intent `exp43b_supplemental_s1.pt` and wake word
`exp43w_wake_s1.pt` (SHA-256 of both ONNX files equal to master's).

| Intent model | Size | p50 ms | p95 ms | RTF p95 | Features ms | Model ms | Wake word, share of 1 core | Wake hop p95 ms | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcm_intent.onnx` (Exp 43b), run 2 | 1,463 KB | 13.88 | 15.19 | 0.0061 | 3.81 | 10.09 | 1.88% | 1.96 | 99.3 |

Run 1 (printed by the deploy script): p50 13.8 ms, p95 14.7 ms, RTF p95 0.0059, 3.8 ms
features + 10.0 ms model, 100 MB peak.

Memory with both services running: 719 MB used of 8,063 MB (7,343 MB available, no swap);
`vcm_listen.py` 99.5 MB RSS, `vcm.home.server` 171.5 MB RSS (after a live test that
played music; it was 47.0 MB in the earlier measurement below).

## Earlier: Experiment 41d / 40b weights (same architectures)

Repo at master `11fc1d5`; intent `exp41d_combo_wide_distill_s0.pt`, wake word
`exp42_wake_me2_s1.pt`.

| Intent model | Size | p50 ms | p95 ms | RTF p95 | Features ms | Model ms | Wake word, share of 1 core | Wake hop p95 ms | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcm_intent.onnx` (Exp 41d), run 2 | 1,463 KB | 13.93 | 15.78 | 0.0063 | 3.75 | 9.98 | 1.85% | 1.90 | 99.6 |
| `vcm_intent.onnx`, run 1 | 1,463 KB | 14.25 | 14.97 | 0.0060 | 3.93 | 10.27 | 1.94% | — | 99.3 |
| `vcm_intent_small.onnx` (Exp 40b) | 722 KB | 11.07 | 12.06 | 0.0048 | 3.93 | 7.21 | 1.98% | 2.01 | 98.3 |

Memory with both services running: 604 MB used of 8,063 MB (7,458 MB available, no swap);
`vcm_listen.py` 95.3 MB RSS, `vcm.home.server` 47.0 MB RSS.

The shipped Experiment 43b model has the same architecture and operations as 41d, and
measured the same within run-to-run noise. The small model (Experiment 43c, same
architecture as 40b) has not been re-timed with its new weights; expect the 40b numbers.

For comparison, the previous intent model (Exp 36, 432 KB) took 9.9 ms per command
(3.7 ms features + 6.2 ms model) on the same Pi.
