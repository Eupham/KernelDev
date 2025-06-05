import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Optional, Dict, Any


class TokenizedDataset(Dataset):
    """Dataset that handles tokenized text data for language modeling."""
    
    def __init__(self, tokenized_data, seq_len=512):
        self.data = tokenized_data
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        # Get sequence and target (next token prediction)
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


class DataBuilder:
    """Handles data loading, preprocessing, and tokenization for training."""
    
    def __init__(
        self,
        dataset_name: str = "allenai/c4",
        dataset_config: str = "en",
        seq_len: int = 512,
        max_samples: Optional[int] = 2000,  # Default to 2000 records
        vocab_size: int = 256  # UTF-8 byte vocabulary
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.vocab_size = vocab_size
        
        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
    
    def _tokenize_text(self, text: str) -> list:
        """Convert text to UTF-8 byte tokens."""
        # Encode text as UTF-8 bytes and convert to list of integers
        return list(text.encode('utf-8'))
    
    def _detokenize_bytes(self, tokens: list) -> str:
        """Convert UTF-8 byte tokens back to text."""
        try:
            # Convert tokens to bytes and decode as UTF-8
            byte_data = bytes(tokens)
            return byte_data.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Error decoding tokens: {e}")
            return f"[DECODE_ERROR: {tokens[:10]}...]"
    
    def load_raw_dataset(self):
        """Load the raw dataset from HuggingFace."""
        print(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        
        try:
            dataset = load_dataset(self.dataset_name, self.dataset_config)
            print(f"Dataset loaded successfully!")
            print(f"Available splits: {list(dataset.keys())}")
            
            # Show dataset info
            for split_name, split_data in dataset.items():
                print(f"{split_name}: {len(split_data)} samples")
                if len(split_data) > 0:
                    print(f"Sample fields: {list(split_data[0].keys())}")
                    print(f"Sample text preview: {str(split_data[0])[:200]}...")
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to a simple text dataset...")
            return self._create_fallback_dataset()
    
    def _create_fallback_dataset(self):
        """Create a simple fallback dataset if the main dataset fails to load."""
        # Create some sample text data
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing practice.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "Natural language processing enables computers to understand and generate human language.",
            "Transformers have revolutionized the field of natural language processing with their attention mechanisms.",
            "GPT models are based on the transformer architecture and are trained on large amounts of text data.",
            "Flash attention is an efficient implementation of the attention mechanism that reduces memory usage.",
            "PyTorch is a popular deep learning framework that provides dynamic computation graphs.",
            "CUDA enables parallel computing on NVIDIA GPUs for accelerated machine learning workloads.",
            "Tokenization is the process of converting text into numerical tokens that models can process.",
        ] * 100  # Repeat to create more data
        
        return {
            'train': [{'text': '\n'.join(sample_texts)}],
            'validation': [{'text': '\n'.join(sample_texts[:50])}],
            'test': [{'text': '\n'.join(sample_texts[50:100])}]
        }
    
    def tokenize_dataset(self, dataset):
        """Tokenize the text data in the dataset."""
        tokenized_data = {}
        
        for split_name, split_data in dataset.items():
            print(f"Tokenizing {split_name} split...")
            
            # Combine all text from this split
            if isinstance(split_data, list):
                # Fallback dataset format
                all_text = ""
                for item in split_data:
                    if isinstance(item, dict) and 'text' in item:
                        all_text += item['text'] + "\n"
                    else:
                        all_text += str(item) + "\n"
            else:
                # HuggingFace dataset format
                all_text = ""
                text_field = 'text' if 'text' in split_data[0] else list(split_data[0].keys())[0]
                
                max_samples = self.max_samples if self.max_samples else len(split_data)
                for i, item in enumerate(split_data):
                    if i >= max_samples:
                        break
                    text_content = item[text_field]
                    if text_content and text_content.strip():  # Skip empty entries
                        all_text += text_content + "\n"
            
            # Tokenize the combined text using UTF-8 bytes
            print(f"Text length for {split_name}: {len(all_text)} characters")
            tokens = self._tokenize_text(all_text)
            print(f"Tokenized to {len(tokens)} byte tokens")
            
            tokenized_data[split_name] = tokens
        
        return tokenized_data
    
    def create_datasets(self):
        """Create PyTorch datasets from the tokenized data."""
        # Load and tokenize raw data
        raw_dataset = self.load_raw_dataset()
        tokenized_data = self.tokenize_dataset(raw_dataset)
        
        # Create PyTorch datasets
        datasets = {}
        for split_name, tokens in tokenized_data.items():
            if len(tokens) > self.seq_len:
                datasets[split_name] = TokenizedDataset(tokens, self.seq_len)
                print(f"{split_name} dataset: {len(datasets[split_name])} samples")
            else:
                print(f"Warning: {split_name} split has insufficient tokens ({len(tokens)}) for sequence length {self.seq_len}")
        
        return datasets
    
    def create_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle_train: bool = True
    ) -> Dict[str, DataLoader]:
        """Create DataLoaders for training, validation, and test sets."""
        datasets = self.create_datasets()
        dataloaders = {}
        
        for split_name, dataset in datasets.items():
            shuffle = shuffle_train if split_name == 'train' else False
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            print(f"{split_name} dataloader: {len(dataloaders[split_name])} batches")
        
        return dataloaders
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.vocab_size
    
    def decode_tokens(self, tokens):
        """Decode tokens back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self._detokenize_bytes(tokens)


def create_data_builder(
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    seq_len: int = 512,
    max_samples: Optional[int] = 2000
) -> DataBuilder:
    """Factory function to create a DataBuilder instance."""
    return DataBuilder(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        max_samples=max_samples
    )


if __name__ == "__main__":
    # Test the data builder
    print("Testing DataBuilder...")
    
    # Create data builder
    data_builder = create_data_builder(
        dataset_name="allenai/c4",
        dataset_config="en",
        seq_len=128,
        max_samples=1000
    )
    
    # Create dataloaders
    dataloaders = data_builder.create_dataloaders(batch_size=4)
    
    # Test a batch
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Sample input tokens: {x[0][:10].tolist()}")
            print(f"Sample target tokens: {y[0][:10].tolist()}")
            
            # Decode sample
            sample_text = data_builder.decode_tokens(x[0][:50])
            print(f"Sample text: {sample_text}")
            break
    
    print("DataBuilder test completed!")
