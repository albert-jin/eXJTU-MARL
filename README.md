# eXJTU-MARL: Efficient Exploration Just-in-time and Replay Trajectory Utilization
The Python implementation of our efficient eXploration Just-in-time and Replay Trajectory Utilization (eXJTU-MARL).

## Overview
eXJTU-MARL is a novel framework designed to enhance exploration and learning efficiency in Multi-Agent Reinforcement Learning (MARL) environments. It addresses two main challenges in MARL:
1. **Insufficient Exploration**: Agent policies often overfit and fail to explore effectively, leading to suboptimal strategies.
2. **Sample Inefficiency**: Mainstream MARL algorithms require a large number of environment interactions due to inefficient learning from replay buffers.

eXJTU-MARL introduces two key mechanisms:
- **Adaptive Policy Resetting Mechanism (APRM)**: Periodically resets parts of the agents' policies to prevent overfitting and promote continuous exploration.
- **Balanced Experience Sampling (BES)**: Selects diverse and informative samples from the experience replay buffer, enhancing the learning efficiency.

## Key Features
- **Adaptive Policy Resetting**: Ensures agents do not get stuck in local optima and keeps exploration dynamic throughout the training process.
- **Balanced Experience Sampling**: Employs state representations (dynamics-based and mutual simulation-based) to prioritize diverse replay samples for better learning.
- **Optimized for Complex Environments**: Extensively tested on StarCraft Multi-Agent Challenge (SMAC), outperforming other baseline MARL algorithms.

## Installation
1. Clone the repository:
git clone https://github.com/albert-jin/eXJTU-MARL

2. Install dependencies:
pip install -r requirements.txt


## Usage
To run the eXJTU-MARL framework, you can simply execute the following command:
python train.py --config=config_file.yaml

This script will initiate training based on the configurations provided.

## Results
eXJTU-MARL has shown superior performance across various environments in SMAC, achieving higher win rates and faster convergence compared to mainstream algorithms like QMIX and VDN.

## Citation
If you use this code in your research, please cite:
@article{jin2024exjtu-marl, title={Enhancing Multi-Agent Reinforcement Learning via Efficient Exploration and Learning: Adaptive Policy Resetting and Balanced Experience Sampling}, author={Weiqiang Jin, Xingwu Tian, Ningwei Wang, Baohai Wu, Bohang Shi, Biao Zhao, Guang Yang}, journal={IEEE}, year={2024} }

## License
This project is licensed under the MIT License - see the LICENSE file for details.
