# Pensieve Retraining
This repository provides a reproducible method for retraining the [Pensieve](http://web.mit.edu/pensieve/) model, including the following improvements based on the original Pensieve code:

- Support for **dynamic entropy weight** $\beta$, i.e., decaying from 1 to 0.1 over $10^5$ iterations. This is implemented by modifying `sim/a3c.py` and `sim\multi_agent.py`. Refer to: [Why the result is not better than MPC? · Issue #11 · hongzimao/pensieve](https://github.com/hongzimao/pensieve/issues/11).
- Train and test Pensieve under **higher video bitrate** (up to 4K resolution encoded at 40Mbps). Specifically,`VIDEO_BIT_RATE`, `REBUF_PENALTY`, and chunk size information in Pensieve, BBA , and RobustMPC are modified. The video ([Big Buck Bunny](https://peach.blender.org/)) is provided in `sim/` and `test/`.
- **Normalize states and rewards** for higher network bandwidth (e.g., in 5G networks) by an order of magnitude. Refer to: [godka/pensieve-5G: Pensieve for 5G datasets](https://github.com/godka/pensieve-5G).
- Carefully split the dataset into training and test sets in a repeatable way.
- Fix some bugs in the original code.

Please see the commits after `0f1aa30389fb1798e10c3828cd6bcbed85672021` (2024-01-13 UTC+8) for details. 

Note: This repository only reports single-video simulation results. Potential issues may exist in the multi-video scenario, emulation, or real-world deployment, where the related code has not been modified . 



### Training and testing
Pensieve's original training and testing procedure remains changed. Specifically


