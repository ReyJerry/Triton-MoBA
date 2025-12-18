import torch
import random
import time
from flash_attn import flash_attn_varlen_func
from triton_moba.efficient_moba import moba_attn_varlen_efficient
from triton_moba.triton_moba import moba_attn_varlen_triton


def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # gen qkv
    q = torch.randn(
        (seqlen, num_q_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )

    # gen cu seqlen
    cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
    cu_seqlen.sort()
    cu_seqlen = [0] + cu_seqlen + [seqlen]
    cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)

    # max_seqlen
    max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])

    return q, k, v, cu_seqlen, max_seqlen.item()


def benchmark_kernel(func, name, args, kwargs, warmup_iters=3, perf_test_iters=10, vo_grad=None):
    """Helper function to benchmark a specific kernel"""
    # Warmup
    for _ in range(warmup_iters):
        o = func(*args, **kwargs)
        if vo_grad is not None:
            torch.autograd.backward(o, vo_grad)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(perf_test_iters):
        o = func(*args, **kwargs)
        if vo_grad is not None:
            torch.autograd.backward(o, vo_grad)
            
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - start_time) / perf_test_iters * 1000
    print(f"{name}: {avg_time:.2f}ms")
    return avg_time


def test_attn_varlen_moba_speed(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk, dtype=torch.bfloat16):
    """Speed test comparing Flash Attention, Efficient MoBA, and Triton MoBA"""
    print(f"\nBenchmark Config: batch={batch} head={head} seqlen={seqlen} chunk={moba_chunk_size} topk={moba_topk}")
    print("-" * 60)

    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    vo_grad = torch.randn_like(q)
    
    # 1. Benchmark Flash Attention (Baseline)
    time_flash = benchmark_kernel(
        flash_attn_varlen_func,
        "Flash Attention (Standard)",
        args=(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen),
        kwargs={"causal": True},
        vo_grad=vo_grad
    )

    # 2. Benchmark Efficient MoBA (PyTorch Baseline)
    time_efficient = benchmark_kernel(
        moba_attn_varlen_efficient,
        "Efficient MoBA (PyTorch)",
        args=(q, k, v, cu_seqlen, max_seqlen),
        kwargs={"moba_chunk_size": moba_chunk_size, "moba_topk": moba_topk},
        vo_grad=vo_grad
    )

    # 3. Benchmark Triton MoBA (Optimized Target)
    time_triton = benchmark_kernel(
        moba_attn_varlen_triton,
        "Triton MoBA (Optimized)",
        args=(q, k, v, cu_seqlen, max_seqlen),
        kwargs={"moba_chunk_size": moba_chunk_size, "moba_topk": moba_topk},
        vo_grad=vo_grad
    )
    
    print("-" * 60)
    print(f"Speedup vs Efficient MoBA (PyTorch): {time_efficient / time_triton:.2f}x")
    print(f"Speedup vs Flash Attention:          {time_flash / time_triton:.2f}x")
    print("-" * 60)


if __name__ == "__main__":
    # Test case with large sequence length to highlight optimization
    test_attn_varlen_moba_speed(batch=1, head=1, seqlen=32768, head_dim=128, moba_chunk_size=512, moba_topk=3)
    print("Benchmark finished")
