# Performance Benchmarking Guide

This guide describes the benchmarking suite for measuring and tracking training performance in KernelDev.

## Overview

The benchmarking suite provides tools to measure:
- **Training Throughput**: Tokens/samples/steps per second
- **Memory Usage**: Component-wise memory profiling
- **GPU Utilization**: Hardware usage efficiency

## Quick Start

### Training Throughput Benchmark

Measure how fast the model trains:

```bash
python benchmarks/training_throughput.py --config config.yaml --steps 100
```

This will:
- Run 100 training steps (after 10 warmup steps)
- Measure tokens/second, samples/second, step time
- Track memory usage
- Save results to `benchmarks/results/throughput_results.json`

### Memory Profile

Analyze memory usage breakdown:

```bash
python benchmarks/memory_profile.py --config config.yaml
```

This will:
- Profile model memory (parameters)
- Profile optimizer memory (Adam states)
- Profile activation memory (forward pass)
- Profile gradient memory (backward pass)
- Track memory timeline over 20 steps
- Save results to `benchmarks/results/memory_profile.json`

## Detailed Usage

### Training Throughput

```bash
python benchmarks/training_throughput.py \
    --config config.yaml \
    --steps 100 \
    --warmup 10 \
    --output benchmarks/results/my_benchmark.json \
    --device cuda
```

**Options**:
- `--config`: Path to YAML configuration file
- `--steps`: Number of benchmark steps (default: 100)
- `--warmup`: Number of warmup steps before measurement (default: 10)
- `--output`: Output path for JSON results (default: `benchmarks/results/throughput_results.json`)
- `--device`: Device to run on (`cuda` or `cpu`, default: `cuda`)

**Output Metrics**:
- Tokens per second (mean, std, min, max, median)
- Samples per second
- Step time in milliseconds
- Memory allocated/reserved (if GPU)
- Peak memory usage

**Example Output**:
```
BENCHMARK RESULTS
================================================================

Device: NVIDIA A100-SXM4-80GB
Memory: 79.2 GB
Compute Capability: 8.0

Throughput:
  Tokens/sec:  45234 ± 1823
  Samples/sec: 11.23 ± 0.45

Timing:
  Step time:   356.42 ± 14.23 ms
  Min:         331.12 ms
  Max:         398.67 ms

Memory:
  Allocated:   4523 MB (avg), 4987 MB (max)
  Reserved:    5120 MB (avg), 5120 MB (max)
  Peak:        5234 MB

Model:
  Parameters:  355,532,800
  Batch size:  4
  Seq length:  1024
```

### Memory Profile

```bash
python benchmarks/memory_profile.py \
    --config config.yaml \
    --output benchmarks/results/my_profile.json \
    --device cuda
```

**Options**:
- `--config`: Path to YAML configuration file
- `--output`: Output path for JSON results (default: `benchmarks/results/memory_profile.json`)
- `--device`: Device to profile (default: `cuda`)

**Output Metrics**:
- Model memory (parameters + overhead)
- Optimizer memory (Adam states)
- Activation memory (forward pass)
- Gradient memory (backward pass)
- Memory timeline over multiple steps
- Peak training memory
- Memory efficiency ratio

**Example Output**:
```
MEMORY PROFILE SUMMARY
================================================================

Device: NVIDIA A100-SXM4-80GB
Total Memory: 79.2 GB

Component Memory:
  Model:         1420.45 MB
  Optimizer:     2840.90 MB
  Activations:   512.34 MB
  Gradients:     823.12 MB

Summary:
  Static (model + optimizer): 4261.35 MB
  Dynamic (activations + gradients): 1335.46 MB
  Estimated total: 5596.81 MB
  Peak training:   5789.23 MB
  Memory efficiency: 1.03x
  GPU utilization: 7.1%
```

## JSON Output Format

### Throughput Results

```json
{
  "config": {...},
  "device": "cuda",
  "device_name": "NVIDIA A100-SXM4-80GB",
  "device_properties": {
    "total_memory_gb": 79.2,
    "multi_processor_count": 108,
    "major": 8,
    "minor": 0
  },
  "metrics": {
    "tokens_per_second": [45234, 45123, ...],
    "samples_per_second": [11.23, 11.19, ...],
    "steps_per_second": [2.81, 2.80, ...],
    "step_times": [0.356, 0.357, ...],
    "memory_allocated_mb": [4523, 4534, ...],
    "memory_reserved_mb": [5120, 5120, ...]
  },
  "summary": {
    "tokens_per_second": {
      "mean": 45234,
      "std": 1823,
      "min": 42341,
      "max": 48765,
      "median": 45123
    },
    ...
  },
  "peak_memory_mb": 5234,
  "timestamp": "2026-01-26T23:42:00.123456"
}
```

### Memory Profile Results

```json
{
  "config": {...},
  "device": "cuda",
  "device_name": "NVIDIA A100-SXM4-80GB",
  "component_memory": {
    "model": {
      "parameters": 355532800,
      "memory_mb": 1420.45,
      "theoretical_mb": 1422.13,
      "overhead_mb": -1.68
    },
    "optimizer": {
      "memory_mb": 2840.90,
      "type": "AdamW"
    },
    "forward_activations": {
      "memory_mb": 512.34,
      "batch_size": 4,
      "seq_len": 1024
    },
    "gradients": {
      "memory_mb": 823.12,
      "peak_allocated_mb": 5789.23
    }
  },
  "memory_timeline": [
    {"step": 0, "allocated_mb": 4523, "reserved_mb": 5120, "max_allocated_mb": 4523},
    ...
  ],
  "peak_training_memory_mb": 5789.23,
  "summary": {
    "total_static_mb": 4261.35,
    "total_dynamic_mb": 1335.46,
    "total_estimated_mb": 5596.81,
    "peak_training_mb": 5789.23,
    "memory_efficiency": 1.03
  },
  "timestamp": "2026-01-26T23:42:00.123456"
}
```

## Comparing Results

To compare benchmark results across different configurations or code changes:

```python
import json

# Load two benchmark results
with open('benchmarks/results/baseline.json', 'r') as f:
    baseline = json.load(f)

with open('benchmarks/results/optimized.json', 'r') as f:
    optimized = json.load(f)

# Compare throughput
baseline_tps = baseline['summary']['tokens_per_second']['mean']
optimized_tps = optimized['summary']['tokens_per_second']['mean']
speedup = optimized_tps / baseline_tps

print(f"Speedup: {speedup:.2f}x")
print(f"Baseline: {baseline_tps:.0f} tokens/s")
print(f"Optimized: {optimized_tps:.0f} tokens/s")
```

## Interpreting Results

### Training Throughput

**Tokens per second**: Higher is better. Typical values:
- Small models (<500M params) on A100: 40,000-80,000 tokens/s
- Medium models (500M-2B params) on A100: 20,000-40,000 tokens/s
- Large models (>2B params) on A100: 5,000-20,000 tokens/s

**Step time**: Lower is better. Should be consistent across steps:
- High variance (>20% std) indicates instability or data loading issues
- Increasing over time indicates memory leak or accumulation

### Memory Usage

**Memory efficiency**: Ratio of actual peak memory to estimated memory:
- 1.0-1.2x: Excellent (minimal overhead)
- 1.2-1.5x: Good (acceptable overhead)
- >1.5x: Poor (investigate memory leaks or inefficiencies)

**Component breakdown**:
- Model: Should match theoretical size (parameters × bytes_per_param)
- Optimizer (AdamW): ~2x model size (stores momentum and variance)
- Activations: Depends on batch size and sequence length
- Gradients: Similar to model size

**GPU utilization**: Percentage of total GPU memory used:
- <10%: Very small model, could increase batch size
- 10-70%: Normal range for most training
- >70%: Approaching memory limits, risk of OOM
- >90%: Very risky, reduce batch size

## Best Practices

1. **Run warmup**: Always use warmup steps to ensure stable measurements
2. **Multiple runs**: Run benchmarks multiple times for statistical significance
3. **Consistent config**: Use the same config for fair comparisons
4. **Clean environment**: Close other GPU processes before benchmarking
5. **Save baselines**: Keep baseline results before optimization work
6. **Document changes**: Note code changes when comparing results

## Troubleshooting

### Out of Memory (OOM)

If benchmarking fails with OOM:
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 2  # Reduce from 4

# Or reduce sequence length
data:
  seq_len: 512  # Reduce from 1024
```

### Slow Performance

If throughput is unexpectedly low:
1. Check GPU utilization with `nvidia-smi`
2. Ensure CUDA is properly installed
3. Verify data loading isn't bottleneck (increase `num_workers`)
4. Check for CPU-GPU transfer overhead
5. Profile with `torch.profiler` for detailed analysis

### Inconsistent Results

If results vary significantly between runs:
1. Increase number of benchmark steps
2. Increase warmup steps
3. Check for background processes
4. Ensure deterministic mode if needed
5. Monitor temperature throttling

## Integration with CI/CD

To track performance regressions automatically:

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on: [pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: |
          python benchmarks/training_throughput.py --steps 50 --output results.json
      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py baseline.json results.json
```

## Future Enhancements

Planned additions to the benchmarking suite:
- [ ] Multi-GPU scaling efficiency
- [ ] End-to-end training time estimation
- [ ] Attention kernel-specific benchmarks
- [ ] Data loading pipeline benchmarks
- [ ] Inference performance benchmarks
- [ ] Comparative analysis tools
- [ ] Automated regression detection
- [ ] Performance visualization dashboard
