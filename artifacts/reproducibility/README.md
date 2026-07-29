# Reproducibility evidence

- `trace_manifest.json` records the source ZIP SHA-256 and the sorted
  per-file size/SHA-256 inventory for 410 training and 100 held-out traces.
  Both filename and content-hash intersections are empty.
- `environment.json` records the exact Python interpreter, platform and
  installed package versions used by the smoke tests and formal runs.

Formal runs use seed 42, 16 synchronous CPU agents, 110,000 updates and a
held-out test every 100 updates. They are started from scratch. A TensorFlow
checkpoint is not treated as a reproducible continuation point because it
does not capture the environment positions, multiprocessing queue state or
all random streams.

The deterministic smoke test compares two independent Windows-spawn runs. It
requires identical central logs, held-out summaries, per-agent trace/chunk
order, state SHA-256 sequence and sampled actions.
