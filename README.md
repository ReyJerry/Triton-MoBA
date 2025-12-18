# Triton-MoBA: High-Performance Mixture of Block Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Triton](https://img.shields.io/badge/Backend-OpenAI%20Triton-blue)](https://triton-lang.org/)

**Triton-MoBA** is a highly optimized implementation of the [MoBA (Mixture of Block Attention)](https://github.com/MoonshotAI/MoBA/tree/master) mechanism using custom OpenAI Triton kernels.

This repository serves as a drop-in replacement for the original PyTorch-based implementation, offering **~2.8x faster training speed** and significantly reduced memory footprint by fusing key operations and minimizing global memory I/O.

## 🚀 Performance

Benchmark conducted on a single **NVIDIA RTX A6000** training a 50M parameter model on a 500M token dataset for 1000 steps.

| Implementation | Training Time | Speedup |
| :--- | :--- | :--- |
| Original (`moba_efficient.py`) | ~85 minutes | 1.0x |
| **Triton Optimized (`moba_triton.py`)** | **~30 minutes** | **~2.8x** |
| Flash Attention | ~13 minutes | ~6.5x |

> **Note**: The optimized implementation maintains exact interface compatibility and autograd correctness with the original code.

## ✨ Key Optimizations

The original implementation relied on standard PyTorch operations which resulted in high memory I/O overheads from materializing large intermediate tensors. This repository replaces those bottlenecks with custom fused kernels:

1.  **Fused Chunk Mean Calculation**:
    * Replaced the memory-intensive `view().mean()` operation.
    * Computes the mean of Key vectors within chunks without materializing intermediate reshaped tensors.

2.  **Fused Gating & Top-K Selection**:
    * Replaces the expensive `torch.einsum` + `torch.topk` pipeline.
    * Fuses **Gating Score Calculation**, **Causal Masking**, and **Top-K Selection** into a single kernel.
    * Avoids instantiating the massive `[Batch, Head, Chunk, Seq]` score matrix, drastically reducing peak memory usage.

3.  **Fused Merge Softmax**:
    * Combines Self-Attention and MoBA-Attention outputs in a single pass using on-chip SRAM.
    * Performs LogSumExp (LSE) reduction and output merging simultaneously, avoiding multiple global memory reads/writes.

4.  **Optimized Backward Pass**:
    * Custom `_gather_moba_backward_inputs_kernel` efficiently gathers gradients and forward activations needed for the sparse attention branch's backward pass.

5.  **Layout Optimization**:
    * Refactored index calculations to work directly with `[Seq, Head, Dim]` layout, removing expensive `rearrange` operations found in the original code.

## 🛠️ Installation

```bash
git clone [https://github.com/ReyJerry/Triton-MoBA.git](https://github.com/ReyJerry/Triton-MoBA.git)
cd Triton-MoBA
pip install -r requirements.txt

```

## 💻 Usage

`Triton-MoBA` is designed as a drop-in replacement. You can import the optimized attention function directly:

```python
import torch
from moba_triton import moba_attn_varlen_triton

# Example inputs (Standard FlashAttention format)
q = torch.randn(seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
cu_seqlens = ... # Cumulative sequence lengths
max_seqlen = ... 

# Call the optimized function
output = moba_attn_varlen_triton(
    q, k, v, 
    cu_seqlens=cu_seqlens, 
    max_seqlen=max_seqlen, 
    moba_chunk_size=1024, 
    moba_topk=2
)

```

## 🧪 Testing & Verification

We provide scripts to verify both correctness (against the original implementation) and performance speedup.

```bash
# Verify correctness (Output consistency check)
python tests/test_triton_moba_attn.py

# Run benchmark (Reproduce speedup results)
python tests/test_triton_moba_speedup.py

```

## 📑 Acknowledgements & Citation

This work is based on the [MoBA](https://github.com/MoonshotAI/MoBA/tree/master) project by Moonshot AI. If you use MoBA in your research, please cite the original paper:

```bibtex
@article{lu2025mobamixtureblockattention,
  author = {Enzhe Lu and Zhejun Jiang and Jingyuan Liu and Yulun Du and Tao Jiang and Chao Hong and Shaowei Liu and Weiran He and Enming Yuan and Yuzhi Wang and Zhiqi Huang and Huan Yuan and Suting Xu and Xinran Xu and    Guokun Lai and Yanru Chen and Huabin Zheng and Junjie Yan and Jianlin Su and Yuxin Wu and Yutao Zhang and Zhilin Yang and Xinyu Zhou and Mingxing Zhang and Jiezhong Qiu},
  title = {MoBA: Mixture of Block Attention for Long-Context LLMs},
  journal={arXiv preprint arXiv:2502.13189},
  year={2025}
}

```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

* Original Code License: Copyright © 2025 Moonshot AI
* Triton Optimization: Copyright © 2025 ReyJerry
