# Deep Reinforcement Learning for Service Function Chain Placement with Graph Attention and Transformer Encoder

## Introduction

This repository contains the codes of paper *Deep Reinforcement Learning for Service Function Chain Placement with Graph Attention and Transformer Encoder*. 

In this paper, we propose a novel DRL approach based on **Proximal Policy Optimization (PPO)**. Our method employs a **Graph Attention Network (GAT)** to capture local topological information and a **Transformer encoder** to model the sequential dependencies within the SFC. Unlike traditional VNF-by-VNF placement strategies, our model generates the complete placement configuration for an entire SFC in a single forward pass. Experimental results demonstrate that our approach achieves performance gains of at least 23.92% in acceptance ratio and 44.18% in reducing exceeded resource usage, and significantly lowers both power consumption and SFC end-to-end latency.

## Requirements

- Python 3.9.21
- CUDA 12.6.0

https://developer.nvidia.com/cuda-12-6-0-download-archive

- PyTorch 2.6.0

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Other required libraries are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Quick Start

You can run the following code to train and evaluate all the agents.

```
python ./main.py
```

To configure the parameters, you can view all available options in `config.py`. 

All trained agent models are stored in `save/model`. The corresponding training and evaluation results, along with performance plots, are saved in `save/result`.
