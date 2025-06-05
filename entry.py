import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from model import GPTModel


class SimpleTextDataset(Dataset):
    """Simple dataset for testing language modeling."""
    
    def __init__(self, text, seq_len=128, vocab_size=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Convert text to tokens (simple character-level tokenization)
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Pad vocab_size if needed
        actual_vocab_size = len(self.chars)
        if actual_vocab_size < vocab_size:
            # Add padding tokens
            for i in range(actual_vocab_size, vocab_size):
                self.idx_to_char[i] = f'<PAD_{i}>'
        
        # Encode text
        self.data = [self.char_to_idx.get(ch, 0) for ch in text]
        
    def __len__(self):
        return max(1, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        # Get sequence and target (next token prediction)
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def create_dummy_dataset(seq_len=128, vocab_size=1000, num_samples=1000):
    """Create a dummy dataset with some patterns for the model to learn."""
    # Create some simple patterns that the model can learn
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world, this is a test sentence. ",
        "Machine learning models can learn patterns in data. ",
        "Artificial intelligence is transforming the world. ",
        "Deep learning uses neural networks with many layers. "
    ]
    
    # Repeat patterns to create a larger dataset
    text = ""
    for _ in range(num_samples // 10):
        for pattern in patterns:
            text += pattern
    
    return SimpleTextDataset(text, seq_len, vocab_size)


def train_model(model, dataloader, optimizer, device, num_epochs=10):
    """Train the model and track loss."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            if loss is not None:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        losses.append(avg_loss)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
    
    return losses


def evaluate_model(model, dataloader, device):
    """Evaluate the model and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def plot_training_curve(losses):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_generation(model, dataset, device, num_tokens=100):
    """Test the model's generation capability."""
    model.eval()
    
    # Start with a random sequence from the dataset
    start_idx = torch.randint(0, len(dataset), (1,))
    x, _ = dataset[start_idx.item()]
    x = x.unsqueeze(0).to(device)  # Add batch dimension
    
    print("Starting sequence:")
    start_text = ''.join([dataset.idx_to_char.get(idx.item(), '?') for idx in x[0]])
    print(f"'{start_text}'")
    
    # Generate new tokens
    generated = model.generate(x, max_new_tokens=num_tokens, temperature=0.8, top_k=50)
    
    print("\nGenerated sequence:")
    generated_text = ''.join([dataset.idx_to_char.get(idx.item(), '?') for idx in generated[0]])
    print(f"'{generated_text}'")


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    vocab_size = 256  # Small vocab for testing
    seq_len = 64
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-3
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'dim': 256,
        'n_layers': 6,
        'n_heads': 8,
        'max_seq_len': 512,
        'mlp_ratio': 4
    }
    
    print("Creating dataset...")
    dataset = create_dummy_dataset(seq_len=seq_len, vocab_size=vocab_size, num_samples=1000)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GPTModel(**model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Evaluate initial loss
    print("\nEvaluating initial model...")
    initial_train_loss = evaluate_model(model, train_loader, device)
    initial_val_loss = evaluate_model(model, val_loader, device)
    print(f"Initial training loss: {initial_train_loss:.4f}")
    print(f"Initial validation loss: {initial_val_loss:.4f}")
    
    # Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    losses = train_model(model, train_loader, optimizer, device, num_epochs)
    
    # Evaluate final loss
    print("\nEvaluating final model...")
    final_train_loss = evaluate_model(model, train_loader, device)
    final_val_loss = evaluate_model(model, val_loader, device)
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Show improvement
    train_improvement = initial_train_loss - final_train_loss
    val_improvement = initial_val_loss - final_val_loss
    print(f"\nTraining loss improvement: {train_improvement:.4f}")
    print(f"Validation loss improvement: {val_improvement:.4f}")
    
    # Plot training curve
    print("\nPlotting training curve...")
    plot_training_curve(losses)
    
    # Test generation
    print("\nTesting text generation...")
    try:
        test_generation(model, dataset, device, num_tokens=50)
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'losses': losses
    }, 'gpt_model_checkpoint.pt')
    
    print("Training completed successfully!")
    print(f"Model demonstrates ability to reduce loss: {train_improvement > 0 and val_improvement > 0}")


if __name__ == "__main__":
    main()
