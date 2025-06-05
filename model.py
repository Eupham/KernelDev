import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from original_kernel import streaming_attention


class RMSNorm(nn.Module):
    """RMS Normalization without learnable weight parameter"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # RMS normalization without weight scaling
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x / rms


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # No bias parameters as requested
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StreamingAttention(nn.Module):
    """Multi-head streaming attention using the custom kernel"""
    def __init__(self, dim, n_heads, context_size=512, back_contexts=4):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.context_size = context_size
        self.back_contexts = back_contexts
        self.scale = self.head_dim ** -0.5
        
        # No bias parameters
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, lens=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use streaming attention kernel with autotuning disabled for stability
        out = streaming_attention(
            q=q,
            k=k,
            v=v,
            lens=lens,
            context_size=self.context_size,
            back_contexts=self.back_contexts,
            sm_scale=self.scale,
            autotune=False,  # Disable autotuning for stability
            return_lse=False,
            prescale_qk=False,  # Disable for better numerical stability
            precision='ieee'  # Use IEEE precision for stability
        )
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with prenorm architecture"""
    def __init__(self, dim, n_heads, context_size=512, back_contexts=4, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        
        # Pre-normalization layers
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        
        # Attention and MLP layers
        self.attn = StreamingAttention(dim, n_heads, context_size, back_contexts)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x, lens=None):
        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x), lens)
        
        # Pre-norm MLP
        x = x + self.mlp(self.mlp_norm(x))
        
        return x


class GPTModel(nn.Module):
    """GPT-style model with streaming attention"""
    def __init__(
        self,
        vocab_size=50257,
        dim=768,
        n_layers=12,
        n_heads=12,
        context_size=512,
        back_contexts=4,
        max_seq_len=2048,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings (no bias)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, context_size, back_contexts, mlp_ratio)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, lens=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, lens)
        
        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits

    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())


def create_model(
    vocab_size=50257,
    dim=512,
    n_layers=8,
    n_heads=8,
    context_size=256,
    back_contexts=2,
    max_seq_len=1024
):
    """Create a GPT model with streaming attention"""
    model = GPTModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        context_size=context_size,
        back_contexts=back_contexts,
        max_seq_len=max_seq_len
    )
    
    # Convert to fp16 as requested
    model = model.half()
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.cuda.amp.autocast():
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")
