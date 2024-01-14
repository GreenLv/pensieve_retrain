# Pensieve Retraining
This repository provides a reproducible method for retraining the [Pensieve](http://web.mit.edu/pensieve/) model, including the following improvements based on the original Pensieve code:

- Support for **dynamic entropy weight** $\beta$, i.e., linearly decaying from 1 to 0.1 over $10^5$ iterations. This is implemented by modifying `sim/a3c.py` and `sim\multi_agent.py`. Refer to: [Why the result is not better than MPC? · Issue #11 · hongzimao/pensieve](https://github.com/hongzimao/pensieve/issues/11).
- Train and test Pensieve under **higher video bitrate** (up to 4K resolution encoded at 40Mbps). Specifically,`VIDEO_BIT_RATE`, `REBUF_PENALTY`, and chunk size information in Pensieve, BBA , and RobustMPC are modified. The video ([Big Buck Bunny](https://peach.blender.org/)) is provided in `sim/` and `test/`.
- **Normalize states and rewards** for higher network bandwidth (e.g., in 5G networks) by an order of magnitude. Refer to: [godka/pensieve-5G: Pensieve for 5G datasets](https://github.com/godka/pensieve-5G).
- Carefully split the dataset into training and test sets in a repeatable way.
- Fix some bugs in the original code.

Please see the commits after `668f7ef1c3be7656d878771591ec93a865f4b1e0` (Jan 13, 2024) for details. 

Note: This repository only reports single-video simulation results. Potential issues may exist in the multi-video scenario, emulation, or real-world deployment, where the related code has not been modified . 



## Changing description

Only files in three folders are changed: 

- `sim/`: dynamic entropy weight; new video; states and rewards normalization.
- `test/`: new video; states and rewards normalization.
- `retrained_info/`: information related to the retrained model, including:
  - `data_preprocess/`: scripts to filter and split the dataset (network traces)
  - `retrained_model/`: retrained model files
  - `training_info/`: training curves about the reward and TD loss; the central agent log
  - `test_results/`: performance of the retrained model versus BBA and RobustMPC



### Training and testing methodology
Pensieve's original training and testing procedure remains unchanged. Specifically, I used a TensorFlow v1.1.0 docker to run the programs. Best practices were provided by the authors:

> `Ubuntu 16.04, Tensorflow v1.1.0, TFLearn v0.3.1 and Selenium v2.39.0`
>
> From: [Issue #12 · hongzimao/pensieve](https://github.com/hongzimao/pensieve/issues/12#issuecomment-345060132)



**Network traces.** Four classes of wireless bandwidth traces are used in training and testing, collected in 3G, 4G, 5G, and Wi-Fi networks. Each class of traces is further divided into several types, depending on the location or mobility. I further filter out traces  (see `retrained_info/data_preprocess/filtered_traces.py`) whose average bandwidth is less than 1.5Mbps (because the lowest video bitrate is 1Mbps), mainly in the Norway FCC dataset. 

|                 | Count | Range of Average Bandwidth (Mbps) |
| --------------- | ----- | --------------------------------- |
| Norway FCC (3G) | 134   | 1.51~4.59                         |
| Lumos4G         | 175   | 7.59~102.43                       |
| Lumos5G         | 121   | 66.49~906.09                      |
| Solis Wi-Fi     | 80    | 7.28~73.16                        |

All traces are publicly available: (however, the unit may be inconsistent)

- Norway FCC (3G)  : [transys-project/pitree/traces.zip](https://github.com/transys-project/pitree/blob/master/traces.zip)
- Lumos4G: [SIGCOMM21-5G/artifact/Video-Streaming/Network-Traces/Lumous5G/4G](https://github.com/SIGCOMM21-5G/artifact/tree/main/Video-Streaming/Network-Traces/Lumous5G/4G)
- Lumos5G: [SIGCOMM21-5G/artifact/Video-Streaming/Network-Traces/Lumous5G/5G](https://github.com/SIGCOMM21-5G/artifact/tree/main/Video-Streaming/Network-Traces/Lumous5G/5G)
- Solis Wi-Fi: [GreenLv/Solis-WiFi-Trace](https://github.com/GreenLv/Solis-WiFi-Trace)



When retraining the Pensieve model, the dataset is split into training and test sets at a ratio of 0.8:0.2. This split is _random and uniform_, depending on each type in each class of all traces. See `retrained_info\data_preprocess\split_trian_test.py` for details.



## Retrained model information

The model is training for 109,900 iterations, taking 7 hours and 44 minutes. 

<p align="left">
    <img src="retrained_info/training_info/training_reward.png" width="60%">
    <img src="retrained_info/training_info/training_loss.png" width="60%">
</p>




## Testing results

BBA ("sim_bb") and RobustMPC ("sim_mpc") are evaluated on the same test set. It can be seen that the retrained Pensieve model ("sim_rl") successfully outperforms these two algorithms, in terms of the average QoE score (5.7% to 28.9% higher). 

<p align="left">
    <img src="retrained_info/test_results/mean_rewards_109900.png" width="40%">
    <img src="retrained_info/test_results/reward_cdf_109900.png" width="40%">
</p>
