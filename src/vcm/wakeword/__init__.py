""""Hey Kiwi" wake-word detection (EXPERIMENTS.md Experiments 33-34).

A separate always-on model, much smaller than the intent model, scores a
sliding 1.5 s window every 0.1 s; when it fires, the device records the
command that follows and hands it to the intent/slot model. Chosen from
a confusability analysis over 62K real command transcripts (EXPERIMENTS.md):
"Kiwi" collides with almost nothing in everyday speech or in this
project's own commands, unlike "Cory" ("increase", "decrease", "create").
"""

WAKE_PHRASE = "hey kiwi"
WINDOW_S = 1.5
HOP_S = 0.1
LABELS = ("not_wake", "wake")
