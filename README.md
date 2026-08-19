<div align="center">

# Micro-Transformer from Scratch
*A minimal, character-level Generative Transformer model built step-by-step in PyTorch.*

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://python.org)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![CUDA Version](https://img.shields.io/badge/CUDA-12.6%2B-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)

⭐ If you find this repository helpful for learning, star it on GitHub!

[Overview](#overview) • [Roadmap](#roadmap) • [Repository Structure](#repository-structure) • [Installation](#installation) • [Usage](#usage)

</div>

---

A clean, transparent, and step-by-step implementation of a Generative pre-trained Transformer (GPT) model from scratch. Built specifically for educational purposes to understand exactly how raw text is tokenized, embedded, processed through attention layers, and used for text generation.

> [!NOTE]
> We deliberately use **character-level tokenization** rather than subword tokenization (like BPE). While this sacrifices linguistic complexity, it keeps the vocabulary small (~65 unique tokens) and the dimensions transparent and easy to debug.

---

## Roadmap

The project is structured incrementally to build up the entire architecture piece-by-piece:

* **Day 1: Input Pipeline & Dataset Preparation** (Complete)
  * Load raw Tiny Shakespeare text
  * Build character vocabulary and mapping dictionaries (`char2int`/`int2char`)
  * Implement `encode` and `decode` functions
  * Convert corpus to PyTorch tensors and split into training (90%) and validation (10%) sets
  * Build random batch generator (`get_batch`) with autoregressive target shift verification
* **Day 2: Attention Mechanism** (Upcoming)
  * Query, Key, and Value ($Q, K, V$) projections
  * Scaled dot-product attention
  * Causal attention masking (autoregressive masking)
  * Multi-Head Attention
* **Day 3: Transformer Block & Normalization** (Upcoming)
  * Layer Normalization (LayerNorm)
  * Feed-Forward Network (FFN)
  * Residual Connections
* **Day 4: Full Model & Output Head** (Upcoming)
  * Token and Position Embeddings
  * Stacked Transformer Blocks
  * Final LayerNorm and Output projection head
* **Day 5: Training Loop & Generation** (Upcoming)
  * Loss calculation and backpropagation
  * Training loop optimizer integration
  * Generative text decoding pipeline

---

## Repository Structure

```text
micro-Transformer-from-scratch/
├── data/                  # Tiny Shakespeare dataset
│   └── input.txt          # Plain text source file
├── experiments/           # Sanity checks and testing scripts
│   └── attention_sanity_check.py
├── src/                   # Source code
│   ├── dataset.py         # Tokenizer, vocabulary, train/val split & batching
│   ├── embeddings.py      # Token and position embeddings
│   └── attention.py       # Self-attention and multi-head attention module
└── README.md              # Project documentation
```

---

## Installation

### Prerequisites
* **Python**: 3.11 or 3.12 (recommended)
* **GPU**: CUDA-compatible NVIDIA GPU (recommended for acceleration, though CPU works for smaller steps)

### Step 1: Clone and Set Up Virtual Environment
```powershell
# Set up Python virtual environment
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch with CUDA
Install PyTorch using the official CUDA index (example for CUDA 12.6):
```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Usage

### Running the Data Pipeline (Day 1)
Execute the dataset script to load Tiny Shakespeare, train-validation split, and verify batch generation:

```powershell
python src/dataset.py
```

### Expected Output
When run, the script prints dataset statistics, verifies the tokenizer, generates a training batch, and asserts the autoregressive next-token target shift:

```text
Number of characters: 1115394
Vocabulary size: 65
Vocabulary:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

Tokenizer test
Original: Hello transformer!
Encoded: [20, 43, 50, 50, 53, 1, 58, 56, 39, 52, 57, 44, 53, 56, 51, 43, 56, 2]
Decoded: Hello transformer!

Data shape: torch.Size([1115394])
Data type: torch.int64
Train data shape: torch.Size([1003854])
Val data shape: torch.Size([111540])

x shape: torch.Size([8, 128])
y shape: torch.Size([8, 128])
```

> [!TIP]
> The target tensor `y` is exactly input `x` shifted one position to the left. The validation test ensures that `x[:, 1:]` is identical to `y[:, :-1]` for all batches.