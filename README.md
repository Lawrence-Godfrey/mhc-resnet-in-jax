# Manifold-Constrained Hyper-Connections (mHC) in JAX

A notebook exploring the evolution of residual connections in deep learning, from standard Residual Connections to DeepSeek's proposed **[Manifold-Constrained Hyper-Connections (mHC)]((https://arxiv.org/abs/2512.24880))**.

Each architecture is implemented using **[JAX NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)**, a JAX-based (experimental) object-oriented neural network library, and trained on CIFAR-100 using a TPU.

The idea here is to introduce the basic concepts of residual connections, JAX, all in one tutorial notebook, while also delving into the more complex research topic of mHC.

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

## Usage

Open `mhc_notebook.ipynb` in Google Colab or Jupyter and run all cells.
