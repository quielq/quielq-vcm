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
played music; 47 MB before any music had played).

`vcm_intent_small.onnx` (Experiment 43c) has not been re-timed with its final weights;
with earlier weights of the same architecture it took 11.1 ms p50 / 12.1 ms p95
(RTF 0.0048, 7.2 ms model). Measurements of earlier models are in
[docs/legacy/EXPERIMENTS.md](../docs/legacy/EXPERIMENTS.md#raspberry-pi-5-earlier-weights).
