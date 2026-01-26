"""
Training Throughput Benchmark

Measures training performance metrics including:
- Tokens per second
- Samples per second
- Steps per second
- GPU utilization
- Memory usage

Usage:
    python benchmarks/training_throughput.py --config config.yaml --steps 100
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPTModel
from data_builder import create_data_builder
from train_loop import TrainingConfig
import yaml


class ThroughputBenchmark:
    """Benchmark training throughput and performance."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = {
            'config': config,
            'device': device,
            'device_name': None,
            'device_properties': {},
            'metrics': {
                'tokens_per_second': [],
                'samples_per_second': [],
                'steps_per_second': [],
                'step_times': [],
                'memory_allocated_mb': [],
                'memory_reserved_mb': [],
                'gpu_utilization': []
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            self.results['device_name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.results['device_properties'] = {
                'total_memory_gb': props.total_memory / 1e9,
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor
            }
    
    def setup_model_and_data(self):
        """Initialize model and data for benchmarking."""
        print("Setting up model and data...")
        
        # Create model
        model_config = self.config['model']
        self.model = GPTModel(
            vocab_size=model_config['vocab_size'],
            dim=model_config['dim'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            causal=model_config['causal']
        ).to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create data builder
        data_config = self.config['data']
        self.data_builder = create_data_builder(
            dataset_name=data_config.get('dataset_name', 'allenai/c4'),
            dataset_config=data_config.get('dataset_config', 'en'),
            seq_len=data_config['seq_len'],
            max_samples=100,  # Small number for benchmark
            shuffle_train=data_config.get('shuffle_train', True),
            num_workers=data_config.get('num_workers', 0)
        )
        
        # Get batch size
        batch_size = self.config['training'].get('batch_size', 4)
        if batch_size is None:
            batch_size = 4
        self.batch_size = batch_size
        
        print(f"Batch size: {self.batch_size}")
        print(f"Sequence length: {data_config['seq_len']}")
        
        # Create optimizer for realistic benchmark
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training'].get('learning_rate', 3e-4)
        )
        
    def benchmark_step(self, tokens: torch.Tensor, task_name: str = 'teacher_forcing') -> Dict[str, float]:
        """Benchmark a single training step."""
        step_start = time.time()
        
        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_allocated_start = torch.cuda.memory_allocated(0) / 1e6
            mem_reserved_start = torch.cuda.memory_reserved(0) / 1e6
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Prepare input
        input_ids = tokens[:, :-1].to(self.device)
        targets = tokens[:, 1:].to(self.device)
        
        # Run forward pass
        logits = self.model(input_ids, task_name=task_name)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Wait for GPU to finish
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - step_start
        
        # Record final memory
        metrics = {'step_time': step_time, 'loss': loss.item()}
        
        if torch.cuda.is_available():
            metrics['memory_allocated_mb'] = torch.cuda.memory_allocated(0) / 1e6
            metrics['memory_reserved_mb'] = torch.cuda.memory_reserved(0) / 1e6
            
        return metrics
    
    def run_benchmark(self, num_steps: int = 100, warmup_steps: int = 10):
        """Run throughput benchmark for specified number of steps."""
        print(f"\n{'='*60}")
        print(f"Running throughput benchmark for {num_steps} steps")
        print(f"Warmup steps: {warmup_steps}")
        print(f"{'='*60}\n")
        
        self.setup_model_and_data()
        
        # Get data iterator
        train_loader = self.data_builder.get_data_loader('train', self.batch_size)
        train_iter = iter(train_loader)
        
        # Warmup
        print("Warming up...")
        for i in range(warmup_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            tokens = batch['tokens']
            _ = self.benchmark_step(tokens)
        
        # Clear memory stats after warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        print(f"\nRunning benchmark for {num_steps} steps...")
        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            tokens = batch['tokens']
            metrics = self.benchmark_step(tokens)
            
            # Record metrics
            step_time = metrics['step_time']
            batch_size_actual = tokens.size(0)
            seq_len = tokens.size(1) - 1  # -1 because we use [:-1]
            
            tokens_per_second = (batch_size_actual * seq_len) / step_time
            samples_per_second = batch_size_actual / step_time
            steps_per_second = 1.0 / step_time
            
            self.results['metrics']['tokens_per_second'].append(tokens_per_second)
            self.results['metrics']['samples_per_second'].append(samples_per_second)
            self.results['metrics']['steps_per_second'].append(steps_per_second)
            self.results['metrics']['step_times'].append(step_time)
            
            if 'memory_allocated_mb' in metrics:
                self.results['metrics']['memory_allocated_mb'].append(metrics['memory_allocated_mb'])
                self.results['metrics']['memory_reserved_mb'].append(metrics['memory_reserved_mb'])
            
            # Print progress
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step + 1}/{num_steps}: "
                      f"{tokens_per_second:.0f} tokens/s, "
                      f"{samples_per_second:.2f} samples/s, "
                      f"{step_time*1000:.2f} ms/step")
        
        # Calculate peak memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(0) / 1e6
            self.results['peak_memory_mb'] = peak_memory_mb
        
        self._compute_summary_statistics()
        self._print_results()
        
    def _compute_summary_statistics(self):
        """Compute summary statistics from collected metrics."""
        metrics = self.results['metrics']
        
        self.results['summary'] = {
            'tokens_per_second': {
                'mean': float(np.mean(metrics['tokens_per_second'])),
                'std': float(np.std(metrics['tokens_per_second'])),
                'min': float(np.min(metrics['tokens_per_second'])),
                'max': float(np.max(metrics['tokens_per_second'])),
                'median': float(np.median(metrics['tokens_per_second']))
            },
            'samples_per_second': {
                'mean': float(np.mean(metrics['samples_per_second'])),
                'std': float(np.std(metrics['samples_per_second'])),
                'min': float(np.min(metrics['samples_per_second'])),
                'max': float(np.max(metrics['samples_per_second'])),
                'median': float(np.median(metrics['samples_per_second']))
            },
            'step_time_ms': {
                'mean': float(np.mean(metrics['step_times']) * 1000),
                'std': float(np.std(metrics['step_times']) * 1000),
                'min': float(np.min(metrics['step_times']) * 1000),
                'max': float(np.max(metrics['step_times']) * 1000),
                'median': float(np.median(metrics['step_times']) * 1000)
            }
        }
        
        if len(metrics['memory_allocated_mb']) > 0:
            self.results['summary']['memory_allocated_mb'] = {
                'mean': float(np.mean(metrics['memory_allocated_mb'])),
                'max': float(np.max(metrics['memory_allocated_mb']))
            }
            self.results['summary']['memory_reserved_mb'] = {
                'mean': float(np.mean(metrics['memory_reserved_mb'])),
                'max': float(np.max(metrics['memory_reserved_mb']))
            }
    
    def _print_results(self):
        """Print benchmark results to console."""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}\n")
        
        if 'device_name' in self.results and self.results['device_name']:
            print(f"Device: {self.results['device_name']}")
            props = self.results['device_properties']
            print(f"Memory: {props['total_memory_gb']:.1f} GB")
            print(f"Compute Capability: {props['major']}.{props['minor']}")
            print()
        
        summary = self.results['summary']
        
        print("Throughput:")
        print(f"  Tokens/sec:  {summary['tokens_per_second']['mean']:.0f} ± {summary['tokens_per_second']['std']:.0f}")
        print(f"  Samples/sec: {summary['samples_per_second']['mean']:.2f} ± {summary['samples_per_second']['std']:.2f}")
        print()
        
        print("Timing:")
        print(f"  Step time:   {summary['step_time_ms']['mean']:.2f} ± {summary['step_time_ms']['std']:.2f} ms")
        print(f"  Min:         {summary['step_time_ms']['min']:.2f} ms")
        print(f"  Max:         {summary['step_time_ms']['max']:.2f} ms")
        print()
        
        if 'memory_allocated_mb' in summary:
            print("Memory:")
            print(f"  Allocated:   {summary['memory_allocated_mb']['mean']:.0f} MB (avg), "
                  f"{summary['memory_allocated_mb']['max']:.0f} MB (max)")
            print(f"  Reserved:    {summary['memory_reserved_mb']['mean']:.0f} MB (avg), "
                  f"{summary['memory_reserved_mb']['max']:.0f} MB (max)")
            if 'peak_memory_mb' in self.results:
                print(f"  Peak:        {self.results['peak_memory_mb']:.0f} MB")
            print()
        
        # Model info
        model_params = sum(p.numel() for p in self.model.parameters())
        print("Model:")
        print(f"  Parameters:  {model_params:,}")
        print(f"  Batch size:  {self.batch_size}")
        print(f"  Seq length:  {self.config['data']['seq_len']}")
        print()
        
        print(f"{'='*60}\n")
    
    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark training throughput')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of benchmark steps')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup steps')
    parser.add_argument('--output', type=str, default='benchmarks/results/throughput_results.json',
                       help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run benchmark on')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run benchmark
    benchmark = ThroughputBenchmark(config, device=args.device)
    benchmark.run_benchmark(num_steps=args.steps, warmup_steps=args.warmup)
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == '__main__':
    main()
