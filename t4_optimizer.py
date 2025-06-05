import torch
import triton
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from model import GPTModel
from original_kernel import flash_attention


@dataclass
class T4Config:
    """Configuration optimized for T4 GPU."""
    tile_q_size: int
    tile_k_size: int
    num_warps: int
    num_stages: int
    shared_memory_usage: int
    max_seq_len: int
    max_batch_size: int
    model_dim: int
    n_heads: int
    throughput: float = 0.0


class T4Optimizer:
    """Optimizes configurations for T4 GPU constraints."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.t4_shared_memory_limit = 65536  # T4 shared memory limit in bytes
        self.safety_margin = 0.9  # Use 90% of available shared memory for safety
        self.max_safe_shared_memory = int(self.t4_shared_memory_limit * self.safety_margin)
        
        # T4 compute capability and specs
        self.t4_sm_count = 40  # T4 has 40 SMs
        self.t4_max_threads_per_sm = 1024
        self.t4_warp_size = 32
        
        print(f"T4 Optimizer initialized:")
        print(f"  Shared memory limit: {self.t4_shared_memory_limit} bytes")
        print(f"  Safe shared memory: {self.max_safe_shared_memory} bytes")
        print(f"  SM count: {self.t4_sm_count}")
        
    def estimate_shared_memory_usage(self, tile_q: int, tile_k: int, head_dim: int, dtype_size: int = 2) -> int:
        """Estimate shared memory usage for given tile sizes."""
        # Approximate shared memory usage based on flash attention requirements
        # This is a simplified estimation
        
        # Q tile: tile_q x head_dim
        q_memory = tile_q * head_dim * dtype_size
        
        # K tile: tile_k x head_dim  
        k_memory = tile_k * head_dim * dtype_size
        
        # V tile: tile_k x head_dim
        v_memory = tile_k * head_dim * dtype_size
        
        # Attention scores: tile_q x tile_k
        scores_memory = tile_q * tile_k * 4  # float32 for scores
        
        # Additional buffers and overhead
        overhead = 2048  # Conservative overhead estimate
        
        total = q_memory + k_memory + v_memory + scores_memory + overhead
        return total
    
    def find_optimal_tile_sizes(self, head_dim: int = 64) -> List[Tuple[int, int, int]]:
        """Find optimal tile sizes that fit within T4 shared memory constraints."""
        print(f"\nSearching for optimal tile sizes (head_dim={head_dim})...")
        
        valid_configs = []
        tile_sizes = [16, 32, 64, 128]  # Reduced range for T4
        warp_options = [2, 4]  # Reduced warps for T4
        
        for tile_q in tile_sizes:
            for tile_k in tile_sizes:
                for num_warps in warp_options:
                    shared_mem = self.estimate_shared_memory_usage(tile_q, tile_k, head_dim)
                    
                    if shared_mem <= self.max_safe_shared_memory:
                        # Calculate theoretical throughput (higher tile sizes generally better)
                        throughput_score = (tile_q * tile_k) / shared_mem
                        valid_configs.append((tile_q, tile_k, num_warps, shared_mem, throughput_score))
                        
                        print(f"  Valid: tile_q={tile_q}, tile_k={tile_k}, warps={num_warps}, "
                              f"shared_mem={shared_mem}B, score={throughput_score:.4f}")
        
        # Sort by throughput score
        valid_configs.sort(key=lambda x: x[4], reverse=True)
        
        if not valid_configs:
            print("  WARNING: No valid configurations found! Using minimal settings.")
            return [(16, 16, 2)]
        
        print(f"  Found {len(valid_configs)} valid configurations")
        return [(config[0], config[1], config[2]) for config in valid_configs[:5]]  # Top 5
    
    def benchmark_attention_kernel(self, config: T4Config, seq_len: int, batch_size: int) -> float:
        """Benchmark flash attention kernel with given configuration."""
        try:
            # Create test tensors
            q = torch.randn(batch_size, config.n_heads, seq_len, config.model_dim // config.n_heads, 
                          device=self.device, dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            
            # Warmup
            for _ in range(3):
                _ = flash_attention(q, k, v, lens=None, causal=True)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            num_trials = 10
            
            for _ in range(num_trials):
                _ = flash_attention(q, k, v, lens=None, causal=True)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_trials
            
            # Calculate throughput (tokens/second)
            total_tokens = batch_size * seq_len
            throughput = total_tokens / avg_time
            
            return throughput
            
        except Exception as e:
            print(f"    Error benchmarking: {e}")
            return 0.0
    
    def search_optimal_sequence_batch(self, base_config: T4Config) -> Tuple[int, int, float]:
        """Search for optimal sequence length and batch size combination."""
        print(f"\nSearching for optimal sequence length and batch size...")
        
        # Define search ranges
        seq_lengths = [64, 128, 256, 512, 1024]
        batch_sizes = [1, 2, 4, 8, 16]
        
        best_throughput = 0.0
        best_seq_len = 128
        best_batch_size = 2
        results = []
        
        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                try:
                    # Check if this combination fits in memory
                    # Rough memory estimation for model + activations
                    model_params = base_config.model_dim * base_config.model_dim * base_config.n_heads * 4  # Rough estimate
                    activation_memory = batch_size * seq_len * base_config.model_dim * 2  # FP16
                    total_memory_mb = (model_params + activation_memory) / (1024 * 1024)
                    
                    # T4 has 16GB memory, but leave room for other allocations
                    if total_memory_mb > 12000:  # 12GB limit for safety
                        continue
                    
                    print(f"  Testing seq_len={seq_len}, batch_size={batch_size} "
                          f"(~{total_memory_mb:.1f}MB)...")
                    
                    # Test with a minimal model
                    test_config = T4Config(
                        tile_q_size=base_config.tile_q_size,
                        tile_k_size=base_config.tile_k_size,
                        num_warps=base_config.num_warps,
                        num_stages=1,
                        shared_memory_usage=base_config.shared_memory_usage,
                        max_seq_len=seq_len,
                        max_batch_size=batch_size,
                        model_dim=base_config.model_dim,
                        n_heads=base_config.n_heads
                    )
                    
                    throughput = self.benchmark_attention_kernel(test_config, seq_len, batch_size)
                    results.append((seq_len, batch_size, throughput, total_memory_mb))
                    
                    print(f"    Throughput: {throughput:.1f} tokens/sec")
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_seq_len = seq_len
                        best_batch_size = batch_size
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"    OOM: seq_len={seq_len}, batch_size={batch_size}")
                    continue
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        # Print results summary
        print(f"\n  Throughput Results:")
        print(f"  {'Seq Len':<8} {'Batch':<6} {'Throughput':<12} {'Memory (MB)':<12}")
        print(f"  {'-'*40}")
        for seq_len, batch_size, throughput, memory in sorted(results, key=lambda x: x[2], reverse=True):
            print(f"  {seq_len:<8} {batch_size:<6} {throughput:<12.1f} {memory:<12.1f}")
        
        return best_seq_len, best_batch_size, best_throughput
    
    def find_optimal_t4_config(self) -> T4Config:
        """Find the optimal configuration for T4 GPU."""
        print("=" * 60)
        print("T4 GPU OPTIMIZATION SEARCH")
        print("=" * 60)
        
        # Start with reasonable defaults for T4
        head_dim = 64  # Common head dimension
        model_dim = 256  # Smaller model for T4
        n_heads = 4
        
        print(f"Target model: dim={model_dim}, heads={n_heads}, head_dim={head_dim}")
        
        # Find optimal tile sizes
        optimal_tiles = self.find_optimal_tile_sizes(head_dim)
        
        if not optimal_tiles:
            raise RuntimeError("No valid tile configurations found for T4!")
        
        # Use the best tile configuration
        best_tile_q, best_tile_k, best_warps = optimal_tiles[0]
        shared_mem = self.estimate_shared_memory_usage(best_tile_q, best_tile_k, head_dim)
        
        print(f"\nSelected tile configuration:")
        print(f"  tile_q_size: {best_tile_q}")
        print(f"  tile_k_size: {best_tile_k}")
        print(f"  num_warps: {best_warps}")
        print(f"  shared_memory_usage: {shared_mem} bytes ({shared_mem/self.t4_shared_memory_limit*100:.1f}% of limit)")
        
        # Create base config
        base_config = T4Config(
            tile_q_size=best_tile_q,
            tile_k_size=best_tile_k,
            num_warps=best_warps,
            num_stages=1,  # Conservative for T4
            shared_memory_usage=shared_mem,
            max_seq_len=128,  # Will be optimized
            max_batch_size=2,  # Will be optimized
            model_dim=model_dim,
            n_heads=n_heads
        )
        
        # Search for optimal sequence length and batch size
        opt_seq_len, opt_batch_size, max_throughput = self.search_optimal_sequence_batch(base_config)
        
        # Update config with optimal values
        final_config = T4Config(
            tile_q_size=best_tile_q,
            tile_k_size=best_tile_k,
            num_warps=best_warps,
            num_stages=1,
            shared_memory_usage=shared_mem,
            max_seq_len=opt_seq_len,
            max_batch_size=opt_batch_size,
            model_dim=model_dim,
            n_heads=n_heads,
            throughput=max_throughput
        )
        
        return final_config
    
    def print_optimal_config(self, config: T4Config):
        """Print the optimal configuration in a nice format."""
        print("\n" + "=" * 60)
        print("OPTIMAL T4 CONFIGURATION FOUND")
        print("=" * 60)
        
        print("\n🔧 KERNEL CONFIGURATION:")
        print(f"  Tile Q Size:      {config.tile_q_size}")
        print(f"  Tile K Size:      {config.tile_k_size}")
        print(f"  Number of Warps:  {config.num_warps}")
        print(f"  Number of Stages: {config.num_stages}")
        print(f"  Shared Memory:    {config.shared_memory_usage} bytes ({config.shared_memory_usage/self.t4_shared_memory_limit*100:.1f}% of T4 limit)")
        
        print("\n🚀 OPTIMAL THROUGHPUT SETTINGS:")
        print(f"  Sequence Length:  {config.max_seq_len}")
        print(f"  Batch Size:       {config.max_batch_size}")
        print(f"  Throughput:       {config.throughput:.1f} tokens/sec")
        
        print("\n🏗️  MODEL CONFIGURATION:")
        print(f"  Model Dimension:  {config.model_dim}")
        print(f"  Number of Heads:  {config.n_heads}")
        print(f"  Head Dimension:   {config.model_dim // config.n_heads}")
        
        print("\n📝 CODE CONFIGURATION:")
        print("```python")
        print("# T4-Optimized Configuration")
        print("data_config = {")
        print(f"    'seq_len': {config.max_seq_len},")
        print("    'max_samples': 2000")
        print("}")
        print("")
        print("model_config = {")
        print("    'vocab_size': 256,")
        print(f"    'dim': {config.model_dim},")
        print(f"    'n_heads': {config.n_heads},")
        print(f"    'max_seq_len': {config.max_seq_len * 2},")  # Allow 2x for generation
        print("    'causal': True")
        print("}")
        print("")
        print(f"batch_size = {config.max_batch_size}")
        print("```")
        
        print("\n🎯 KERNEL OPTIMIZATION SETTINGS:")
        print("```python")
        print("# Add these to original_kernel.py for T4 optimization")
        print(f"MIN_TILE_SIZE = {min(config.tile_q_size, config.tile_k_size)}")
        print(f"MAX_TILE_SIZE = {max(config.tile_q_size, config.tile_k_size)}")
        print(f"DEFAULT_NUM_WARPS = {config.num_warps}")
        print("```")
        
        print("=" * 60)


def run_t4_optimization():
    """Run the T4 optimization process."""
    print("Starting T4 GPU optimization...")
    
    # Check if we're on CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. This optimization is for T4 GPU.")
        return None
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    
    if "T4" not in gpu_name:
        print(f"Warning: This optimization is designed for T4 GPU, but detected {gpu_name}")
        print("Results may not be optimal for your hardware.")
    
    optimizer = T4Optimizer()
    
    try:
        optimal_config = optimizer.find_optimal_t4_config()
        optimizer.print_optimal_config(optimal_config)
        return optimal_config
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_t4_optimization()
