# Manifold-Constrained Hyper-Connections (mHC)

A Google Colab-compatible notebook exploring the evolution of skip connections in deep learning, from standard Residual Connections to DeepSeek's proposed **Manifold-Constrained Hyper-Connections (mHC)**.

## Overview

This notebook implements and compares three architectures:

| Architecture | Skip Connection | Stability |
|--------------|-----------------|-----------|
| **ResNet** | Identity: $x + F(x)$ | ✅ Stable |
| **Hyper-Connections** | Learned: $H^{res} x + F(x)$ | ⚠️ Unstable at scale |
| **mHC** | Constrained: $P_M(H^{res}) x + F(x)$ | ✅ Stable + Expressive |

## Key Concepts

- **Residual Connections**: The foundation of modern LLMs (GPT, Llama, etc.)
- **Hyper-Connections**: More expressive but prone to gradient explosion
- **mHC**: Projects mixing matrices onto the Birkhoff Polytope using Sinkhorn-Knopp iteration

## Tech Stack

- **JAX NNX**: Object-oriented neural network API for JAX
- **Ray Data/Train**: Distributed data loading and training
- **CIFAR-100**: Dataset for benchmarking

## Usage

Open `mhc_notebook.ipynb` in Google Colab or Jupyter and run all cells.

## References

- [mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880)](https://arxiv.org/abs/2512.24880)
