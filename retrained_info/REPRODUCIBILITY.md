# Corrected normalized beta=1 reference

Published: 2026-07-30

This repository publishes only the corrected normalized beta=1 reference
checkpoint. The five non-normalized beta models are not published here.

## Protocol

- Training baseline: `87abd1b2caf511c21e9bc001df675f35c6bf0cda`
- Formal execution commit: `d5a6571afa7dc68a1497ec6330dc3e3cdfa5389f`
- Seed: 42
- CPU A3C agents: 16
- Updates: 110,000
- Held-out interval: 100
- From scratch: yes
- Training traces: 410
- Held-out traces: 100
- Trace manifest SHA-256:
  `c06e9a51d9931585ba3926bbd1c951e0e0d1e659f96adab8e920766b1b9a4989`

The actor and critic sixth branches in both training and test graphs read
`inputs[:, 5:6, -1]`. Training and held-out testing use the same normalized
state/reward setting.

## Environment

- Python 3.7.16
- TensorFlow 1.15.2
- TFLearn 0.3.2
- NumPy 1.17.2
- protobuf 3.20.3
- Windows 10 build 26100

## Checkpoint

Prefix:
`retrained_info/retrained_model/beta-1_normalized_ep_110000.ckpt`

| Component | SHA-256 |
|---|---|
| data | `7b6c8ecce307747ad0b49fb75a8a24673c9dcc174d2809f319a774ec7cc25084` |
| index | `6012b69a0305484150dfc7fb8b5a36ed8916b1ee1fb461f9d23ab2f9afd2e7c0` |
| meta | `1e32056c306178400a38023ebe93b2287b9ac5c236d06f6c6aa9e4cd7a8740fc` |

`reproduction_manifest.json` is the machine-readable authority and also
records the exact environment inventory, suite-manifest hash, and hashes of
the removed erroneous checkpoint.

## Verification

The six-model suite completed all 110,000 updates without NaN/Inf. The
published model passed TensorFlow restore, finite/normalized actor
probabilities, deterministic single-session smoke tests, and component hash
closure.
