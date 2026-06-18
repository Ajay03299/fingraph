# Design decisions

This document records the engineering trade-offs behind FinGraph. The goal is to
make the reasoning auditable — in a real financial-crime setting, *why* a system
works the way it does matters as much as the fact that it does.

## Why synthetic data (and how its limits are handled)

Real labelled laundering data is scarce and sensitive. A seeded synthetic
generator gives full ground truth, perfect reproducibility, and the freedom to
plant specific typologies. The honest cost is that detecting patterns you planted
risks looking circular, and the synthetic fraud base rate is far higher than
reality (often well under 1%).

These limits are addressed rather than hidden: the roadmap validates the same
pipeline on a public AML benchmark at realistic base rates, and reports how
metrics degrade as fraud becomes rarer. Reported numbers use a fixed seed so they
are reproducible.

## Why IsolationForest for detection

The detection layer is unsupervised, mirroring reality where clean fraud labels
rarely exist at training time. IsolationForest isolates outliers in few random
splits, needs almost no tuning, handles mixed-scale features, and — importantly —
its behaviour is interpretable enough to support an explanation layer. A neural
autoencoder would be harder to justify to a reviewer and far harder to explain to
an investigator.

## Why NetworkX and propagation, not a GNN

A graph neural network is the flashier choice, but for an *investigation* tool the
priorities are explainability, reproducibility, and operability. A GNN's learned
representations are difficult to translate into the concrete reasons a compliance
analyst needs ("flagged because it sits on a cycle with three known-risky
accounts").

Instead, structural signals (fan-in, cycle membership, path position) are computed
explicitly with NetworkX, and risk is propagated across the graph with a damped
diffusion in the spirit of personalised PageRank. This captures the network nature
of laundering while keeping every signal nameable. A GNN branch is on the roadmap
as a benchmarked comparison, not a replacement — the point being to show the
simpler, explainable method was chosen deliberately.

## Why these evaluation metrics

Accuracy is misleading for rare events: flagging nobody scores high and catches
nothing. FinGraph reports ROC-AUC (ranking quality), average precision
(rarity-aware), and precision@k (the metric an investigator working a queue feels
directly).

## Threshold calibration

The reason-code thresholds (e.g. top-percentile fan-in) are calibrated relative to
the population rather than hard-coded, so "extreme" means extreme *for this
dataset*. They are tuned for the synthetic data and would be re-calibrated against
a production distribution. They are deliberately conservative to keep the evidence
list focused on genuinely notable signals.

## Architecture: one pipeline, thin surfaces

All analysis lives in an importable `fingraph` package built around a single
`Pipeline` object constructed once and held in memory. The API and dashboard are
thin readers over it. This keeps logic out of the UI layer, makes the system
testable, and means a request never triggers a full recomputation.
