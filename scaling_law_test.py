import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
from pathlib import Path
import time
import json
from tqdm import tqdm

# Import our custom modules
from model import GPTModel
from data_builder import create_data_builder
from train_loop import TrainingConfig

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class LearningRateScalingTest:
    def __init__(self, config_path='KernelDev/config.yaml', batch_size=16):
        self.config = load_config(config_path)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters from config.yaml
        self.model_dim = self.config['model']['dim']
        self.n_layers = self.config['model']['n_layers']
        self.n_heads = self.config['model']['n_heads']
        self.vocab_size = self.config['model']['vocab_size']
        self.seq_len = self.config['data']['seq_len']
        
        # Setup data
        self.setup_data()
        
        # Learning rates to test (logarithmic scale)
        self.learning_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        
        # Batch counts to test
        self.batch_counts = [10, 100, 1000]
        
        # Results storage
        self.results = {}
        
        # Output directory
        self.output_dir = Path("scaling_law_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def setup_data(self):
        """Set up data for training."""
        print("Setting up data...")
        # Calculate max samples needed based on largest batch count test
        max_batch_count = max(self.batch_counts)
        # Add a 20% buffer to ensure we have enough samples
        max_samples_needed = max_batch_count * self.batch_size * 1.2
        print(f"Setting max_samples to {int(max_samples_needed)} based on largest batch count ({max_batch_count})")
        
        self.data_builder = create_data_builder(
            dataset_name=self.config['data'].get('dataset_name', 'allenai/c4'),
            dataset_config=self.config['data'].get('dataset_config', 'en'),
            seq_len=self.seq_len,
            max_samples=int(max_samples_needed),
            max_eval_tokens=self.config['data'].get('max_eval_tokens', 50000)
        )
        
        # Create datasets using the proper method
        datasets = self.data_builder.create_datasets()
        
        if 'train' not in datasets or not datasets['train']:
            raise RuntimeError("Failed to create training dataset")
        
        # Create DataLoader with fixed batch size
        self.train_loader = torch.utils.data.DataLoader(
            datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def setup_model(self):
        """Create and initialize the model."""
        print(f"Setting up model with dim={self.model_dim}, layers={self.n_layers}, heads={self.n_heads}...")
        model = GPTModel(
            vocab_size=self.vocab_size,
            dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_seq_len=self.seq_len,
            mlp_ratio=self.config['model'].get('mlp_ratio', 4),
            causal=self.config['model'].get('causal', True)
        ).to(self.device)
        return model
    
    def test_learning_rate(self, model, learning_rate, num_batches):
        """Test a specific learning rate for a specific number of batches."""
        print(f"Testing learning_rate={learning_rate}, batches={num_batches}")
        
        # Reset model weights to ensure fair comparison
        model.apply(self._reset_parameters)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        # Training loop
        model.train()
        losses = []
        
        # Use tqdm for progress tracking
        iter_loader = iter(self.train_loader)
        for batch_idx in tqdm(range(num_batches), desc=f"LR={learning_rate}, Batches={num_batches}"):
            try:
                # Get batch data (cycling through dataloader if needed)
                try:
                    x, y = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(self.train_loader)
                    x, y = next(iter_loader)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    y.view(-1)
                )
                
                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}")
                break
        
        # Return metrics
        return {
            'mean_loss': np.mean(losses),
            'final_loss': losses[-1] if losses else float('inf'),
            'min_loss': np.min(losses) if losses else float('inf'),
            'losses': losses
        }
    
    def _reset_parameters(self, module):
        """Reset model parameters for fair comparison between runs."""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    
    def run_experiments(self):
        """Run all learning rate scaling experiments."""
        self.results = {}
        
        # For each batch count
        for batch_count in self.batch_counts:
            self.results[batch_count] = {}
            
            # Create a fresh model
            model = self.setup_model()
            
            # Test each learning rate
            for lr in self.learning_rates:
                try:
                    result = self.test_learning_rate(model, lr, batch_count)
                    self.results[batch_count][lr] = result
                except Exception as e:
                    print(f"Error testing LR={lr}, batches={batch_count}: {e}")
                    self.results[batch_count][lr] = {"error": str(e)}
            
            # Save intermediate results
            self.save_results(f"intermediate_results_batches_{batch_count}.json")
        
        # Save final results
        self.save_results("final_results.json")
        
        # Plot results
        self.plot_results()
    
    def save_results(self, filename):
        """Save results to a JSON file."""
        # Convert learning rates from float to strings for JSON serialization
        serializable_results = {}
        
        for batch_count, lr_results in self.results.items():
            serializable_results[str(batch_count)] = {}
            for lr, metrics in lr_results.items():
                serializable_results[str(batch_count)][str(lr)] = {
                    k: v if not isinstance(v, list) else v[:10]  # Only store first 10 loss values
                    for k, v in metrics.items()
                }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def plot_results(self):
        """Plot learning rate scaling results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot final loss vs learning rate for each batch count
        for i, batch_count in enumerate(self.batch_counts):
            lr_values = []
            final_losses = []
            
            for lr, metrics in self.results[batch_count].items():
                if 'final_loss' in metrics:
                    lr_values.append(lr)
                    final_losses.append(metrics['final_loss'])
            
            if lr_values:
                axes[i].semilogx(lr_values, final_losses, '-o', linewidth=2)
                axes[i].set_xlabel('Learning Rate')
                axes[i].set_ylabel('Final Loss')
                axes[i].set_title(f'Batch Count: {batch_count}')
                axes[i].grid(True, which="both", ls="-")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_rate_scaling.png")
        plt.close()
        
        # Plot learning curves for each batch count
        for batch_count in self.batch_counts:
            plt.figure(figsize=(10, 6))
            
            for lr, metrics in self.results[batch_count].items():
                if 'losses' in metrics and metrics['losses']:
                    plt.plot(metrics['losses'], label=f'LR: {lr}')
            
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'Learning Curves for {batch_count} Batches')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f"learning_curves_{batch_count}_batches.png")
            plt.close()
    
    def find_optimal_learning_rate(self):
        """Find the optimal learning rate based on results."""
        optimal_lrs = {}
        
        for batch_count in self.batch_counts:
            best_lr = None
            best_loss = float('inf')
            
            for lr, metrics in self.results[batch_count].items():
                if 'final_loss' in metrics and metrics['final_loss'] < best_loss:
                    best_loss = metrics['final_loss']
                    best_lr = lr
            
            if best_lr is not None:
                optimal_lrs[batch_count] = {
                    'learning_rate': best_lr,
                    'final_loss': best_loss
                }
        
        # Save optimal learning rates
        with open(self.output_dir / "optimal_learning_rates.json", 'w') as f:
            json.dump(optimal_lrs, f, indent=2)
        
        print("Optimal Learning Rates:")
        for batch_count, info in optimal_lrs.items():
            print(f"Batch Count: {batch_count}, Optimal LR: {info['learning_rate']}, Loss: {info['final_loss']:.6f}")
        
        return optimal_lrs


def parse_args():
    parser = argparse.ArgumentParser(description='Learning Rate Scaling Law Test')
    parser.add_argument('--config', type=str, default='KernelDev/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size to use for testing (default: 16)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Running Learning Rate Scaling Test with:")
    print(f"- Config file: {args.config}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Will test batch counts: [10, 100, 1000]")
    print(f"- Learning rates to test: [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]")
    
    start_time = time.time()
    
    # Run the scaling law test
    test = LearningRateScalingTest(
        config_path=args.config,
        batch_size=args.batch_size
    )
    test.run_experiments()
    
    # Find optimal learning rates
    optimal_lrs = test.find_optimal_learning_rate()
    
    # Print summary
    print("\n" + "="*50)
    print("Learning Rate Scaling Test Complete")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print("="*50)
    
    # Print model size info
    print(f"\nModel Configuration:")
    print(f"- Dimension: {test.model_dim}")
    print(f"- Layers: {test.n_layers}")
    print(f"- Heads: {test.n_heads}")
    print(f"- Total parameters: ~{test.model_dim * test.n_layers * test.model_dim * 4 / 10**6:.2f}M")
    
    print("\nResults saved in: scaling_law_results/")
