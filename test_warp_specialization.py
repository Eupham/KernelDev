#!/usr/bin/env python3
"""
Test script for H100 warp specialization in flash attention kernel.
Tests performance improvements and correctness of the warp-specialized implementation.
"""

import torch
import time
import numpy as np
from original_kernel import flash_attention, flash_attention_reference, enable_warp_specialization

def test_warp_specialization_correctness():
    """Test that warp specialization produces correct results"""
    print("Testing warp specialization correctness...")
    
    # Test configurations
    configs = [
        (2, 4, 32, 64),   # Small
        (4, 8, 128, 64),  # Medium  
        (2, 4, 256, 128), # Large
    ]
    
    for B, H, T, D in configs:
        print(f"Testing config: B={B}, H={H}, T={T}, D={D}")
        
        # Create test inputs
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        
        # Test with warp specialization (autotune=True enables it on H100)
        out_specialized = flash_attention(q, k, v, autotune=True, causal=True)
        
        # Test with regular implementation
        out_regular = flash_attention(q, k, v, autotune=False, causal=True)
        
        # Test with reference implementation
        out_ref, _, _ = flash_attention_reference(q, k, v, causal=True)
        
        # Check correctness
        diff_specialized = torch.max(torch.abs(out_specialized - out_ref)).item()
        diff_regular = torch.max(torch.abs(out_regular - out_ref)).item()
        
        print(f"  Max diff (specialized): {diff_specialized:.6f}")
        print(f"  Max diff (regular): {diff_regular:.6f}")
        
        # Allow small numerical differences
        tolerance = 1e-2  # Relaxed tolerance for fp16
        assert diff_specialized < tolerance, f"Specialized kernel too different: {diff_specialized}"
        assert diff_regular < tolerance, f"Regular kernel too different: {diff_regular}"
    
    print("✓ Correctness tests passed!")

def benchmark_warp_specialization():
    """Benchmark warp specialization performance"""
    print("\nBenchmarking warp specialization performance...")
    
    if not enable_warp_specialization():
        print("Warp specialization not available (requires H100)")
        return
    
    # Benchmark configuration
    B, H, T, D = 4, 8, 512, 128
    warmup_iters = 10
    bench_iters = 50
    
    print(f"Benchmark config: B={B}, H={H}, T={T}, D={D}")
    print(f"Warmup iterations: {warmup_iters}, Benchmark iterations: {bench_iters}")
    
    # Create test inputs
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)  
    v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    
    def benchmark_kernel(kernel_fn, name, **kwargs):
        # Warmup
        for _ in range(warmup_iters):
            _ = kernel_fn(q, k, v, **kwargs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            _ = kernel_fn(q, k, v, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"{name}: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
        return avg_time
    
    # Benchmark different implementations
    time_regular = benchmark_kernel(flash_attention, "Regular kernel", autotune=False)
    time_autotune = benchmark_kernel(flash_attention, "Autotune kernel", autotune=True)  
    time_ref = benchmark_kernel(lambda q,k,v,**kw: flash_attention_reference(q,k,v)[0], "Reference PyTorch")
    
    # Calculate speedups
    speedup_vs_regular = time_regular / time_autotune
    speedup_vs_ref = time_ref / time_autotune
    
    print(f"\nSpeedup vs regular: {speedup_vs_regular:.2f}x")
    print(f"Speedup vs reference: {speedup_vs_ref:.2f}x")

def test_memory_usage():
    """Test memory usage with warp specialization"""
    print("\nTesting memory usage...")
    
    B, H, T, D = 2, 4, 1024, 128
    
    # Create test inputs
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test with warp specialization
    _ = flash_attention(q, k, v, autotune=True)
    memory_specialized = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test with regular implementation  
    _ = flash_attention(q, k, v, autotune=False)
    memory_regular = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Memory usage (specialized): {memory_specialized:.1f} MB")
    print(f"Memory usage (regular): {memory_regular:.1f} MB")
    print(f"Memory overhead: {((memory_specialized - memory_regular) / memory_regular * 100):.1f}%")

def main():
    """Main test function"""
    print("H100 Warp Specialization Test Suite")
    print("=" * 50)
    
    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {capability}")
        print(f"Warp specialization available: {enable_warp_specialization()}")
    else:
        print("CUDA not available")
        return
    
    try:
        # Run tests
        test_warp_specialization_correctness()
        benchmark_warp_specialization()
        test_memory_usage()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
