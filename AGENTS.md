# Repository guidance

This repository contains a Pensieve-derived CPU A3C training and simulation
implementation. The active retraining code is under `sim/`. The `test/`
directory contains the standalone evaluation graph and legacy baselines.
`retrained_info/` preserves the historical data split and the superseded
reference artifacts.

## Retraining workflow

1. Run `python scripts/prepare_traces.py` from the repository root.
2. Run `python scripts/smoke_reproducibility.py` twice-checking the Windows
   spawn path before any formal training.
3. Run `python scripts/run_training_suite.py` for the six formal models.

All training arguments must be explicit. In particular, never change module
globals to select beta, normalization, seed, trace directories, output
directories, agent count, or epoch count. Windows child processes use `spawn`
and must receive the complete configuration.

Run scripts from the repository root. Code must resolve data and video-size
paths explicitly rather than relying on an inherited current directory.
Formal checkpoints are always trained from scratch: TensorFlow checkpoints do
not capture environment position, multiprocessing queues, or all RNG state and
therefore are not accepted as reproducible continuation points.

## Generated artifacts

`work/` and `artifacts/runs/` are generated locally and are not source files.
Do not commit full non-normalized training runs or transient checkpoints here.
Only the repaired, normalized beta=1 reference checkpoint and its closed
provenance summary belong in this repository. Historical artifact hashes must
remain documented even when a superseded model is removed from an active path.

## Relationship to Solis

`E:\GitHub\Solis-code` is a separate repository. It owns the five
non-normalized beta=1..5 checkpoints used by the parameter study, the
normalized beta=1 compatibility checkpoint, all QoE/P.1203 measurements, and
paper evidence. The standard eight-algorithm comparison uses non-normalized
beta=1. A best-of-beta result is reported separately as a tuned performance
upper bound and must never silently replace that default.

Do not push either repository unless the user explicitly asks.
