# VCM benchmark - quielq - 20261003-094831

Wake word: **Hey Kiwi** - trials: 202 with the wake word + 16 without - shuffle seed: 87436 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/kiwi.log

Mic check (Pi input default): signal-to-noise 18.9 dB, laptop speech -14.5 dBFS, room noise -33.4 dBFS - weak

## At a glance

|                                   | overall      | real voice   | synthetic voice |
|-----------------------------------|--------------|--------------|-----------------|
| intent accuracy (19)              | 90.6%        | 86.5%        | 94.3%           |
| command accuracy (93)             | 87.6%        | 80.2%        | 94.3%           |
| false accept (out of scope fired) | 18.8% (3/16) | 20.0% (2/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 7.0%         | 9.3%         | 5.0%            |
| false wake (no wake word, fired)  | 0.0% (0/16)  | 0.0% (0/9)   | 0.0% (0/7)      |
| slot exact                        | 94.1%        | 85.7%        | 100.0%          |
| latency p95                       | 0.69 s       | 0.67 s       | 0.69 s          |

**Pi:** real-time factor 0.010 (p95 0.020), inference 16 ms, CPU temp max 69.4 C, runtime CPU 3% mean, runtime RAM 110 MB peak, 146 MFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 90.6%                | 87.6%                 |
| balanced accuracy                                 | 89.6%                | 88.1%                 |
| precision (macro)                                 | 93.7%                | 94.0%                 |
| recall (macro)                                    | 89.6%                | 88.1%                 |
| F1 (macro)                                        | 90.8%                | 89.1%                 |
| F2 (macro)                                        | 89.9%                | 88.2%                 |
| false accept rate (OOS fired)                     | 18.8%                | 18.8%                 |
| false reject rate (in-scope silent/rejected)      | 7.0%                 | 7.0%                  |
| misfire rate (wrong command fired)                | 1.6%                 | 4.8%                  |
| accuracy 95% CI                                   | [86-94%]             | [82-91%]              |
| false accept 95% CI                               | [7-43%] (3/16)       | [7-43%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 90.6% [86-94%] | 86.5% [78-92%] | 94.3% [88-97%]  |
| balanced accuracy              | 89.6%          | 85.7%          | 93.2%           |
| F1 (macro)                     | 90.8%          | 86.6%          | 93.9%           |
| F2 (macro)                     | 89.9%          | 85.7%          | 93.2%           |
| false accept rate              | 18.8% (3/16)   | 20.0% (2/10)   | 16.7% (1/6)     |
| false reject rate              | 7.0%           | 9.3%           | 5.0%            |
| misfire rate                   | 1.6%           | 3.5%           | 0.0%            |
| **93 commands** accuracy       | 87.6%          | 80.2%          | 94.3%           |
| balanced accuracy              | 88.1%          | 80.2%          | 94.5%           |
| F1 (macro)                     | 89.1%          | 77.0%          | 93.9%           |
| F2 (macro)                     | 88.2%          | 78.5%          | 94.2%           |
| misfire rate                   | 4.8%           | 10.5%          | 0.0%            |
| slot exact (intent right)      | 94.1% (n=101)  | 85.7% (n=42)   | 100.0% (n=59)   |
| latency p50 / p95              | 0.59 / 0.69 s  | 0.55 / 0.67 s  | 0.60 / 0.69 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/9)     | 0.0% (0/7)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n   | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|-----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 16  | 87.5%  | 82.5 min       | 0.092          | 0.075         | 0.078     |
| BRIGHTNESS      | 18  | 94.4%  | 2.2 %          | 0.028          | 0.017         | 0.016     |
| COLOR           | 16  | 93.8%  | -              | -              | 0.062         | 0.062     |
| CREATE_REMINDER | 17  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 17  | 88.2%  | 0.5 deg        | 0.059          | 0.059         | 0.059     |
| TIMER           | 17  | 100.0% | 0.0 s          | 0.000          | 0.422         | 0.488     |
| ALL             | 101 | 94.1%  | -              | 0.044          | 0.106         | 0.117     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 8063.0 MB, Debian GNU/Linux 12 (bookworm), kernel 6.12.109+rpt-rpi-2712, Python 3.11.2
- packages: numpy 1.24.2

| metric                                      | mean / p95 / max            |
|---------------------------------------------|-----------------------------|
| response latency (command end -> Pi output) | 0.299 / 0.689 / 2.132 s     |
| latency p50 / p99                           | 0.586 / 1.744 s             |
| inference time (Pi-reported)                | 15.5 / 21.0 / 24.0 ms       |
| real-time factor (infer / audio window)     | 0.010 / 0.020 / 0.034       |
| CPU temperature                             | 57.0 / 59.5 / 69.4 C        |
| CPU use, whole Pi                           | 5.2 / 23.5 / 100.0 %        |
| CPU use, your runtime process               | 2.9 / 4.0 / 7.9 %           |
| RAM (RSS), your runtime process             | 107.3 / 109.8 / 109.9 MB    |
| RAM used, whole Pi                          | 2389.1 / 2441.4 / 2499.6 MB |
| CPU clock                                   | 1809 / 2400 / 2400 MHz      |
| load average (1 min)                        | 0.28 / 0.59 / 0.93          |
| runtime CPU-seconds per second of speech    | 0.290                       |
| runtime CPU share of wall time              | 2.9%                        |
| throttling flags seen                       | none                        |
| test wall time                              | 59.6 min                    |
| model parameters                            | 370,464                     |
| model size                                  | 1.43 MB                     |
| model FLOPs per inference                   | 146 MFLOP                   |
| effective GFLOP/s (FLOPs / mean infer time) | 9.37                        |

## Most frequent confusions

**intent level:** REJECT -> TEMPERATURE (2); COLOR -> REJECT (2); ALARM -> REJECT (2); LIGHT_OFF -> REJECT (1); PLAY_MUSIC -> NEXT (1); LIGHT_ON -> REJECT (1); LIGHT_OFF -> LIGHT_ON (1); CREATE_REMINDER -> REJECT (1); REJECT -> LIGHT_OFF (1); TIME -> REJECT (1)

**command level:** Brightness 60 percent -> Brightness 20 percent (1); Lights out -> REJECT (1); Set the temperature to 18 degrees -> Temperature 22 degrees (1); Temperature 22 degrees -> Temperature 18 degrees (1); Play music -> Next song (1); REJECT -> Temperature 18 degrees (1); Set color to Blue -> REJECT (1); Lights on -> REJECT (1); Set an alarm for 8:00 AM -> REJECT (1); Shut off the lights -> Lights on (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 88.9%  | 94.1%  | 90.9%  |
| BRIGHTNESS      | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| CALL            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| COLOR           | 18 | 100.0%    | 88.9%  | 94.1%  | 90.9%  |
| CREATE_REMINDER | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| LIGHT_OFF       | 6  | 80.0%     | 66.7%  | 72.7%  | 69.0%  |
| LIGHT_ON        | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| PAUSE           | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PLAY_MUSIC      | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| REJECT          | 16 | 50.0%     | 81.2%  | 61.9%  | 72.2%  |
| STOP            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| TEMPERATURE     | 18 | 89.5%     | 94.4%  | 91.9%  | 93.4%  |
| TIME            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TIMER           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| VOLUME_DOWN     | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_UP       | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
