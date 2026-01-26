"""
Memory Profiling Benchmark

Profiles memory usage during training including:
- Peak memory allocation
- Memory by component (model, optimizer, activations, gradients)
- Memory timeline during training
- Memory efficiency metrics

Usage:
    python benchmarks/memory_profile.py --config config.yaml
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import gc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPTModel
from data_builder import create_data_builder
import yaml


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = {
            'config': config,
            'device': device,
            'device_name': None,
            'device_properties': {},
            'memory_timeline': [],
            'component_memory': {},
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            self.results['device_name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.results['device_properties'] = {
                'total_memory_gb': props.total_memory / 1e9,
                'total_memory_mb': props.total_memory / 1e6
            }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_mb': torch.cuda.memory_allocated(0) / 1e6,
            'reserved_mb': torch.cuda.memory_reserved(0) / 1e6,
            'max_allocated_mb': torch.cuda.max_memory_allocated(0) / 1e6,
            'max_reserved_mb': torch.cuda.max_memory_reserved(0) / 1e6
        }
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def profile_model_memory(self):
        """Profile memory used by model parameters."""
        print("Profiling model memory...")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.reset_peak_stats()
        initial_mem = self.get_memory_stats()
        
        # Create model
        model_config = self.config['model']
        model = GPTModel(
            vocab_size=model_config['vocab_size'],
            dim=model_config['dim'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            causal=model_config['causal']
        ).to(self.device)
        
        model_mem = self.get_memory_stats()
        model_size = model_mem['allocated_mb'] - initial_mem.get('allocated_mb', 0)
        
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        self.results['component_memory']['model'] = {
            'parameters': param_count,
            'memory_mb': model_size,
            'theoretical_mb': param_bytes / 1e6,
            'overhead_mb': model_size - param_bytes / 1e6
        }
        
        print(f"  Parameters: {param_count:,}")
        print(f"  Memory: {model_size:.2f} MB")
        print(f"  Theoretical: {param_bytes/1e6:.2f} MB")
        
        return model
    
    def profile_optimizer_memory(self, model):
        """Profile memory used by optimizer states."""
        print("\nProfiling optimizer memory...")
        
        initial_mem = self.get_memory_stats()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training'].get('learning_rate', 3e-4)
        )
        
        optimizer_mem = self.get_memory_stats()
        optimizer_size = optimizer_mem['allocated_mb'] - initial_mem['allocated_mb']
        
        self.results['component_memory']['optimizer'] = {
            'memory_mb': optimizer_size,
            'type': 'AdamW'
        }
        
        print(f"  Memory: {optimizer_size:.2f} MB")
        
        return optimizer
    
    def profile_forward_pass_memory(self, model, batch_size: int, seq_len: int):
        """Profile memory used during forward pass."""
        print("\nProfiling forward pass memory...")
        
        self.reset_peak_stats()
        
        # Create dummy input
        tokens = torch.randint(0, self.config['model']['vocab_size'], 
                              (batch_size, seq_len), device=self.device)
        
        initial_mem = self.get_memory_stats()
        
        # Forward pass
        with torch.no_grad():
            logits = model(tokens[:, :-1])
        
        forward_mem = self.get_memory_stats()
        activation_size = forward_mem['allocated_mb'] - initial_mem['allocated_mb']
        
        self.results['component_memory']['forward_activations'] = {
            'memory_mb': activation_size,
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        print(f"  Activations: {activation_size:.2f} MB")
        print(f"  Batch size: {batch_size}, Seq len: {seq_len}")
        
        return tokens
    
    def profile_backward_pass_memory(self, model, optimizer, tokens):
        """Profile memory used during backward pass."""
        print("\nProfiling backward pass memory...")
        
        self.reset_peak_stats()
        initial_mem = self.get_memory_stats()
        
        # Forward pass
        optimizer.zero_grad()
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        forward_mem = self.get_memory_stats()
        
        # Backward pass
        loss.backward()
        
        backward_mem = self.get_memory_stats()
        gradient_size = backward_mem['max_allocated_mb'] - forward_mem['allocated_mb']
        
        self.results['component_memory']['gradients'] = {
            'memory_mb': gradient_size,
            'peak_allocated_mb': backward_mem['max_allocated_mb']
        }
        
        print(f"  Gradients: {gradient_size:.2f} MB")
        print(f"  Peak allocated: {backward_mem['max_allocated_mb']:.2f} MB")
    
    def profile_training_timeline(self, model, optimizer, num_steps: int = 20):
        """Profile memory usage over multiple training steps."""
        print(f"\nProfiling memory timeline ({num_steps} steps)...")
        
        # Get data
        data_config = self.config['data']
        batch_size = self.config['training'].get('batch_size', 4)
        if batch_size is None:
            batch_size = 4
        
        data_builder = create_data_builder(
            dataset_name=data_config.get('dataset_name', 'allenai/c4'),
            dataset_config=data_config.get('dataset_config', 'en'),
            seq_len=data_config['seq_len'],
            max_samples=100,
            shuffle_train=data_config.get('shuffle_train', True),
            num_workers=data_config.get('num_workers', 0)
        )
        
        train_loader = data_builder.get_data_loader('train', batch_size)
        train_iter = iter(train_loader)
        
        self.reset_peak_stats()
        
        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            tokens = batch['tokens'].to(self.device)
            
            # Training step
            optimizer.zero_grad()
            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            
            # Record memory
            mem_stats = self.get_memory_stats()
            self.results['memory_timeline'].append({
                'step': step,
                'allocated_mb': mem_stats['allocated_mb'],
                'reserved_mb': mem_stats['reserved_mb'],
                'max_allocated_mb': mem_stats['max_allocated_mb']
            })
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1}: {mem_stats['allocated_mb']:.0f} MB allocated")
        
        # Final peak memory
        final_stats = self.get_memory_stats()
        self.results['peak_training_memory_mb'] = final_stats['max_allocated_mb']
        print(f"\n  Peak training memory: {final_stats['max_allocated_mb']:.2f} MB")
    
    def run_profile(self):
        """Run complete memory profiling."""
        print(f"\n{'='*60}")
        print("MEMORY PROFILING")
        print(f"{'='*60}\n")
        
        if not torch.cuda.is_available():
            print("CUDA not available - memory profiling requires GPU")
            return
        
        # Profile each component
        model = self.profile_model_memory()
        optimizer = self.profile_optimizer_memory(model)
        
        batch_size = self.config['training'].get('batch_size', 4)
        if batch_size is None:
            batch_size = 4
        seq_len = self.config['data']['seq_len']
        
        tokens = self.profile_forward_pass_memory(model, batch_size, seq_len)
        self.profile_backward_pass_memory(model, optimizer, tokens)
        self.profile_training_timeline(model, optimizer)
        
        self._compute_summary()
        self._print_results()
    
    def _compute_summary(self):
        """Compute summary statistics."""
        comp_mem = self.results['component_memory']
        
        total_static = (
            comp_mem.get('model', {}).get('memory_mb', 0) +
            comp_mem.get('optimizer', {}).get('memory_mb', 0)
        )
        
        total_dynamic = (
            comp_mem.get('forward_activations', {}).get('memory_mb', 0) +
            comp_mem.get('gradients', {}).get('memory_mb', 0)
        )
        
        self.results['summary'] = {
            'total_static_mb': total_static,
            'total_dynamic_mb': total_dynamic,
            'total_estimated_mb': total_static + total_dynamic,
            'peak_training_mb': self.results.get('peak_training_memory_mb', 0),
            'memory_efficiency': self.results.get('peak_training_memory_mb', 0) / 
                                (total_static + total_dynamic) if (total_static + total_dynamic) > 0 else 0
        }
    
    def _print_results(self):
        """Print profiling results."""
        print(f"\n{'='*60}")
        print("MEMORY PROFILE SUMMARY")
        print(f"{'='*60}\n")
        
        if 'device_name' in self.results and self.results['device_name']:
            print(f"Device: {self.results['device_name']}")
            print(f"Total Memory: {self.results['device_properties']['total_memory_gb']:.1f} GB\n")
        
        comp_mem = self.results['component_memory']
        
        print("Component Memory:")
        print(f"  Model:         {comp_mem.get('model', {}).get('memory_mb', 0):.2f} MB")
        print(f"  Optimizer:     {comp_mem.get('optimizer', {}).get('memory_mb', 0):.2f} MB")
        print(f"  Activations:   {comp_mem.get('forward_activations', {}).get('memory_mb', 0):.2f} MB")
        print(f"  Gradients:     {comp_mem.get('gradients', {}).get('memory_mb', 0):.2f} MB")
        print()
        
        summary = self.results['summary']
        print("Summary:")
        print(f"  Static (model + optimizer): {summary['total_static_mb']:.2f} MB")
        print(f"  Dynamic (activations + gradients): {summary['total_dynamic_mb']:.2f} MB")
        print(f"  Estimated total: {summary['total_estimated_mb']:.2f} MB")
        print(f"  Peak training:   {summary['peak_training_mb']:.2f} MB")
        print(f"  Memory efficiency: {summary['memory_efficiency']:.2f}x")
        
        if 'device_properties' in self.results:
            total_mem = self.results['device_properties']['total_memory_mb']
            usage_pct = (summary['peak_training_mb'] / total_mem) * 100
            print(f"  GPU utilization: {usage_pct:.1f}%")
        
        print()
        print(f"{'='*60}\n")
    
    def save_results(self, output_path: str):
        """Save profiling results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Profile memory usage')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='benchmarks/results/memory_profile.json',
                       help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to profile')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Run profiling
    profiler = MemoryProfiler(config, device=args.device)
    profiler.run_profile()
    
    # Save results
    profiler.save_results(args.output)


if __name__ == '__main__':
    main()
