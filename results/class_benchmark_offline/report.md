# VCM benchmark - student - None

Wake word: **None** - trials: 202 with the wake word + 186 without - shuffle seed: None - connection: offline dress rehearsal - holdout: None

## At a glance

|                                   | overall      | real voice   | synthetic voice |
|-----------------------------------|--------------|--------------|-----------------|
| intent accuracy (19)              | 96.0%        | 91.7%        | 100.0%          |
| command accuracy (93)             | 93.6%        | 86.5%        | 100.0%          |
| false accept (out of scope fired) | 18.8% (3/16) | 30.0% (3/10) | 0.0% (0/6)      |
| false reject (command ignored)    | 1.1%         | 2.3%         | 0.0%            |
| false wake (no wake word, fired)  | 0.0% (0/186) | 0.0% (0/86)  | 0.0% (0/100)    |
| slot exact                        | 95.3%        | 89.1%        | 100.0%          |
| latency p95                       | -            | -            | -               |

**Pi:** real-time factor 0.003 (p95 0.005), inference 5 ms

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 96.0%                | 93.6%                 |
| balanced accuracy                                 | 95.5%                | 94.5%                 |
| precision (macro)                                 | 95.5%                | 95.6%                 |
| recall (macro)                                    | 95.5%                | 94.5%                 |
| F1 (macro)                                        | 95.4%                | 94.3%                 |
| F2 (macro)                                        | 95.4%                | 94.3%                 |
| false accept rate (OOS fired)                     | 18.8%                | 18.8%                 |
| false reject rate (in-scope silent/rejected)      | 1.1%                 | 1.1%                  |
| misfire rate (wrong command fired)                | 1.6%                 | 4.3%                  |
| accuracy 95% CI                                   | [92-98%]             | [89-96%]              |
| false accept 95% CI                               | [7-43%] (3/16)       | [7-43%]               |
| false wake rate (command without wake word fired) | 0.0% [0-2%] (0/186)  | 0.0% [0-2%] (0/186)   |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice  |
|--------------------------------|----------------|----------------|------------------|
| clips (with wake word)         | 202            | 96             | 106              |
| **19 intents** accuracy        | 96.0% [92-98%] | 91.7% [84-96%] | 100.0% [97-100%] |
| balanced accuracy              | 95.5%          | 91.0%          | 100.0%           |
| F1 (macro)                     | 95.4%          | 90.6%          | 100.0%           |
| F2 (macro)                     | 95.4%          | 90.7%          | 100.0%           |
| false accept rate              | 18.8% (3/16)   | 30.0% (3/10)   | 0.0% (0/6)       |
| false reject rate              | 1.1%           | 2.3%           | 0.0%             |
| misfire rate                   | 1.6%           | 3.5%           | 0.0%             |
| **93 commands** accuracy       | 93.6%          | 86.5%          | 100.0%           |
| balanced accuracy              | 94.5%          | 88.2%          | 100.0%           |
| F1 (macro)                     | 94.3%          | 86.3%          | 100.0%           |
| F2 (macro)                     | 94.3%          | 87.2%          | 100.0%           |
| misfire rate                   | 4.3%           | 9.3%           | 0.0%             |
| slot exact (intent right)      | 95.3% (n=107)  | 89.1% (n=46)   | 100.0% (n=61)    |
| latency p50 / p95              | -              | -              | -                |
| false wake rate (no wake word) | 0.0% (0/186)   | 0.0% (0/86)    | 0.0% (0/100)     |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n   | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|-----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 18  | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 18  | 94.4%  | 2.2 %          | 0.028          | 0.017         | 0.016     |
| COLOR           | 17  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 18  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 18  | 77.8%  | 1.1 deg        | 0.139          | 0.113         | 0.111     |
| TIMER           | 18  | 100.0% | 0.0 s          | 0.000          | 0.426         | 0.491     |
| ALL             | 107 | 95.3%  | -              | 0.042          | 0.094         | 0.104     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max       |
|---------------------------------------------|------------------------|
| response latency (command end -> Pi output) | -                      |
| latency p50 / p99                           | -                      |
| inference time (Pi-reported)                | 5.4 / 6.2 / 29.0 ms    |
| real-time factor (infer / audio window)     | 0.003 / 0.005 / 0.021  |
| CPU temperature                             | -                      |
| CPU use, whole Pi                           | -                      |
| CPU use, your runtime process               | -                      |
| RAM (RSS), your runtime process             | -                      |
| RAM used, whole Pi                          | -                      |
| CPU clock                                   | -                      |
| load average (1 min)                        | -                      |
| runtime CPU-seconds per second of speech    | -                      |
| runtime CPU share of wall time              | -                      |
| throttling flags seen                       | none                   |
| test wall time                              | 0.0 min                |

## Most frequent confusions

**intent level:** LIGHT_OFF -> LIGHT_ON (1); COLOR -> REJECT (1); PLAY_MUSIC -> STOP (1); REJECT -> PLAY_MUSIC (1); REJECT -> TEMPERATURE (1); REJECT -> ALARM (1); NEXT -> REJECT (1); PLAY_MUSIC -> NEXT (1)

**command level:** Shut off the lights -> Lights on (1); Set color to Blue -> REJECT (1); Play some music -> Stop (1); Temperature 22 degrees -> Temperature 18 degrees (1); REJECT -> Play music (1); Brightness 60 percent -> Brightness 20 percent (1); REJECT -> Temperature 18 degrees (1); Set the temperature to 18 degrees -> Temperature 22 degrees (1); Temperature 18 degrees -> Temperature 22 degrees (1); REJECT -> Alarm 8:00 AM (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 94.7%     | 100.0% | 97.3%  | 98.9%  |
| BRIGHTNESS      | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| CALL            | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| COLOR           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| LIGHT_OFF       | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| LIGHT_ON        | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| PAUSE           | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PLAY_MUSIC      | 6  | 80.0%     | 66.7%  | 72.7%  | 69.0%  |
| REJECT          | 16 | 86.7%     | 81.2%  | 83.9%  | 82.3%  |
| STOP            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| TEMPERATURE     | 18 | 94.7%     | 100.0% | 97.3%  | 98.9%  |
| TIME            | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| TIMER           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_DOWN     | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_UP       | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
