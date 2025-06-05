#!/usr/bin/env python3
"""
Entry script to test the GPT model's ability to reduce loss during training.
This script creates a simple training loop to demonstrate loss reduction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from model import create_model


def generate_dummy_data(vocab_size, seq_len, batch_size, num_batches):
    """Generate dummy training data"""
    data = []
    for _ in range(num_batches):
        # Create random sequences
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Targets are the same sequence shifted by 1
        targets = torch.cat([input_ids[:, 1:], torch.randint(0, vocab_size, (batch_size, 1))], dim=1)
        data.append((input_ids, targets))
    return data


def train_step(model, batch, optimizer, scaler, device):
    """Single training step"""
    input_ids, targets = batch
    input_ids, targets = input_ids.to(device), targets.to(device)
    
    optimizer.zero_grad()
    
    with autocast():
        logits = model(input_ids)
        # Calculate cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def calculate_perplexity(losses):
    """Calculate perplexity from average loss"""
    avg_loss = np.mean(losses)
    return np.exp(avg_loss)


def main():
    # Configuration
    config = {
        'vocab_size': 1000,  # Smaller vocab for faster training
        'dim': 256,          # Smaller model for testing
        'n_layers': 4,
        'n_heads': 4,
        'context_size': 128,
        'back_contexts': 2,
        'max_seq_len': 512,
        'batch_size': 4,
        'seq_len': 128,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'num_batches_per_epoch': 20
    }
    
    print("=" * 60)
    print("GPT Model with Streaming Attention - Loss Reduction Test")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU. Performance will be limited.")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        context_size=config['context_size'],
        back_contexts=config['back_contexts'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.get_num_params()
    print(f"Model created with {num_params:,} parameters")
    print(f"Model memory footprint: ~{num_params * 2 / 1e6:.1f} MB (fp16)")
    
    # Setup optimizer and scaler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    
    # Generate training data
    print(f"\nGenerating {config['num_batches_per_epoch']} batches of dummy data...")
    train_data = generate_dummy_data(
        config['vocab_size'],
        config['seq_len'],
        config['batch_size'],
        config['num_batches_per_epoch']
    )
    
    # Training loop
    print("\nStarting training...")
    print("Epoch | Avg Loss | Perplexity | Time (s)")
    print("-" * 45)
    
    all_losses = []
    epoch_times = []
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        epoch_losses = []
        
        model.train()
        for batch_idx, batch in enumerate(train_data):
            loss = train_step(model, batch, optimizer, scaler, device)
            epoch_losses.append(loss)
            all_losses.append(loss)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        avg_loss = np.mean(epoch_losses)
        perplexity = calculate_perplexity(epoch_losses)
        
        print(f"{epoch+1:5d} | {avg_loss:8.4f} | {perplexity:10.2f} | {epoch_time:7.2f}")
    
    print("-" * 45)
    
    # Final statistics
    initial_loss = np.mean(all_losses[:config['num_batches_per_epoch']])
    final_loss = np.mean(all_losses[-config['num_batches_per_epoch']:])
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    
    print(f"\nTraining Summary:")
    print(f"  Initial loss (epoch 1): {initial_loss:.4f}")
    print(f"  Final loss (epoch {config['num_epochs']}): {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.2f}%")
    print(f"  Average time per epoch: {np.mean(epoch_times):.2f}s")
    
    # Test if model is learning
    if loss_reduction > 5:  # At least 5% reduction
        print(f"✅ SUCCESS: Model shows learning capability with {loss_reduction:.2f}% loss reduction!")
    else:
        print(f"⚠️  WARNING: Limited learning observed. Loss reduction: {loss_reduction:.2f}%")
    
    # Generate a sample
    print(f"\nGenerating sample text...")
    model.eval()
    with torch.no_grad():
        sample_input = torch.randint(0, config['vocab_size'], (1, 20)).to(device)
        
        with autocast():
            logits = model(sample_input)
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_tokens = torch.topk(probs, 5)
        
        print(f"Input tokens: {sample_input[0].cpu().tolist()}")
        print(f"Top 5 predicted next tokens: {top_tokens.indices.cpu().tolist()}")
        print(f"Their probabilities: {top_tokens.values.cpu().tolist()}")
    
    # Model architecture summary
    print(f"\nModel Architecture Summary:")
    print(f"  - Using streaming attention with context_size={config['context_size']}")
    print(f"  - Back contexts: {config['back_contexts']}")
    print(f"  - Pre-norm architecture with RMSNorm (no weight param)")
    print(f"  - SwiGLU activation in feedforward layers")
    print(f"  - No bias parameters throughout the model")
    print(f"  - FP16 precision")
    
    print("\n" + "=" * 60)
    print("Training completed!")


if __name__ == "__main__":
    main()
