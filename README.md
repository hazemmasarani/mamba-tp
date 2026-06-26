# Mamba-TP

Tensor Parallelism implementation for the Hugging Face Mamba model  
`state-spaces/mamba-2.8b-hf`

## Overview

Mamba-TP is an experimental Tensor Parallelism implementation for the Mamba architecture using the Hugging Face model:

`state-spaces/mamba-2.8b-hf`

The project focuses on distributed inference and benchmarking of Mamba by splitting the model across devices over the intermediate size dimension.

The same input is broadcast to all devices, while each device computes a partition of the intermediate representation locally.

Two communication points are introduced in every layer:

- **SSM Parameter Synchronization**
  - All-Reduce sum over the locally computed SSM parameters:
  - B, C, and Δ (delta)
- **Hidden State Synchronization**
  - All-Reduce sum over the hidden states output
  - Produces the final layer output for the following layer

This design enables distributed execution while preserving the correctness of the original model computation.

## Model Architecture

Example:
Mamba-TP Architecture

!\[Mamba-TP Architecture\](assets/mamba_tp_architecture.png)

## Features

- Tensor Parallelism for state-spaces/mamba-2.8b-hf
- Distributed inference execution
- Parallelization over intermediate (hidden) dimension
- Two All-Reduce communication points per layer
- Forward execution with and without cache
- Runtime logging
- Rank-0 output saving
- Tensor Parallel benchmark sweep script

## Project Structure

.  
├── input  
│ ├── gen_input.py  
│ └── input_ids.pt  
├── LICENSE  
├── load_weights.py  
├── log  
│ └── tp.log  
├── model  
│ ├── modeling_mamba.py  
│ └── state_dict.pt  
├── README.md  
├── results  
│ ├── logits_dist_4_30.pt  
│ ├── logits_dist_4_513.pt  
│ └── logits_dist_4.pt  
├── run_cache.py  
├── run.py  
└── run_tp_sweep.sh

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- CUDA-enabled GPUs (recommended)

Install dependencies:

pip install torch transformers

## Setup

### 1\. Load Model Weights

Before running inference, load the model weights:

python load_weights.py

This script downloads and processes the weights for:

state-spaces/mamba-2.8b-hf

The processed state dictionary is saved to:

model/state_dict.pt

## Input Preparation

You can either:

### Option 1 - Use Existing Input IDs

Place your input tensor in:

input/input_ids.pt

### Option 2 - Generate Random Input IDs

`python input/gen_input.py -batch_size 2 -seq_len 128`

## Running Inference

### Forward Pass Without Cache

Runs standard forward cycles without cache reuse.

python run.py

### Forward Pass With Cache

Runs cached autoregressive inference.

python run_cache.py

Behavior:

- First cycle runs without cache
- Remaining 512 cycles run with cache
- Total cycles: 513

This mode is useful for benchmarking autoregressive decoding performance.

## Logging

Runtime information for each cycle is stored in:

log/tp.log

## Results

Inference outputs are stored in:

results/

Only rank 0 saves the output tensors.

Example outputs:

logits_dist_4.pt  
logits_dist_4_30.pt  
logits_dist_4_513.pt

## Benchmarking

Run tensor parallel benchmark sweeps using:

bash run_tp_sweep.sh

This script automates benchmarking across tensor parallel configurations.

## Tensor Parallel Design

### Parallelization Strategy

The model is partitioned across devices using the intermediate (hidden) dimension.

Each device computes:

- Local projections
- Local SSM computations
- Local hidden states

The input tensor is shared across all devices.

### Communication Points

Each layer introduces two communication stages:

#### 1\. SSM Synchronization

After computing local SSM parameters:

B_local  
C_local  
Δ_local

An All-Reduce sum is performed to obtain:

B  
C  
Δ

globally across all devices.

#### 2\. Hidden State Synchronization

After computing the local hidden state outputs:

h_local

An All-Reduce sum is performed to generate the final hidden state:

h

which is then passed to the next layer.

## Notes

- This project is intended for experimentation and benchmarking
- Distributed execution assumes a properly configured PyTorch distributed environment
- NCCL backend is recommended for GPU execution
- Only rank 0 stores the final output tensors

## Future Improvements

Potential future extensions:

- Pipeline Parallelism
- Multi-node support
- Quantization
- FlashAttention integration
- Performance visualization tools
- Training support

## License

This project is licensed under the MIT License.