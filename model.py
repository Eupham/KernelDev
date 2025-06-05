import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming entry.py is in python path or same directory for this import to work
# If entry.py is not found during execution, PYTHONPATH might need adjustment
# or a more robust import mechanism.
from entry import causal_attention

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The 'dim' parameter in RMSNorm typically refers to the feature dimension
        # over which normalization is performed (usually the last dimension).
        # nn.Parameter for learnable scale (gamma) is often used, but omitted here as per spec.
        # self.weight = nn.Parameter(torch.ones(dim)) # Example if gamma was used

    def forward(self, x):
        # Calculate the root mean square over the last dimension
        # x: (B, T, dim)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize
        normalized_x = x / rms
        # Apply learnable scale if it were used:
        # return self.weight * normalized_x
        return normalized_x

class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, in_dim, bias=bias)
        # For SwiGLU, sometimes a beta parameter is applied to w1 or w2's output.
        # F.silu(x) is x * sigmoid(x)

    def forward(self, x):
        # x: (B, T, in_dim)
        # silu_x1 is silu(w1(x))
        silu_x1 = F.silu(self.w1(x))
        # x2 is w2(x)
        x2 = self.w2(x)
        # Element-wise product
        hidden_state = silu_x1 * x2 # (B, T, hidden_dim)
        # Output projection
        return self.w_out(hidden_state) # (B, T, in_dim)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Attention projections (Q, K, V from a single projection)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection for attention
        self.attn_out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Feed-forward network (using SwiGLU)
        # ff_hidden_dim is the intermediate "expanded" dimension in the FFN
        self.ffn = SwiGLU(embed_dim, ff_hidden_dim, bias=False)

        # Normalization layers
        self.norm1 = RMSNorm(embed_dim) # Norm before attention
        self.norm2 = RMSNorm(embed_dim) # Norm before FFN

        # SM Scale for attention (1 / sqrt(head_dim))
        self.sm_scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, lens: torch.Tensor = None): # x: (B, T, embed_dim)
        # Pre-normalization for attention
        x_norm1 = self.norm1(x)

        # Get Q, K, V projections
        qkv = self.qkv_proj(x_norm1) # (B, T, embed_dim * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # each (B, T, embed_dim)

        # Reshape for multi-head attention
        # Current: (B, T, embed_dim)
        # Target for causal_attention: (B, num_heads, T, head_dim)
        B, T, _ = q.shape # q.shape is (B, T, embed_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        # Perform causal attention
        # causal_attention returns (output, LSE_if_requested)
        # We typically don't need LSE for inference or standard training loop unless for specific analysis
        attn_output, _ = causal_attention(
            q, k, v,
            lens=lens, # Pass lens if provided
            sm_scale=self.sm_scale,
            return_lse=False # Set to True if LSE is needed by a custom loss/backward pass
        )
        # attn_output shape: (B, num_heads, T, head_dim)

        # Concatenate heads and project back to embed_dim
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, embed_dim)
        print(f"DEBUG: attn_output shape before transpose: {attn_output.shape}, numel: {attn_output.nelement()}")
        transposed_attn = attn_output.transpose(1, 2)
        print(f"DEBUG: attn_output shape after transpose: {transposed_attn.shape}, numel: {transposed_attn.nelement()}")
        contiguous_attn = transposed_attn.contiguous()
        print(f"DEBUG: attn_output shape after contiguous: {contiguous_attn.shape}, numel: {contiguous_attn.nelement()}")
        print(f"DEBUG: Target view parameters: B={B}, T={T}, self.embed_dim={self.embed_dim}")
        # Original line was: attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attn_output = contiguous_attn.view(B, T, self.embed_dim) # Use intermediate variable
        attn_output = self.attn_out_proj(attn_output) # (B, T, embed_dim)

        # First residual connection
        x = x + attn_output

        # Pre-normalization for FFN
        x_norm2 = self.norm2(x)

        # FFN
        ffn_output = self.ffn(x_norm2) # (B, T, embed_dim)

        # Second residual connection
        x = x + ffn_output

        return x

class LiteGPTModel(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_hidden_dim: int,
                 vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Initialize weights - good practice
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # Should not happen given bias=False in lm_head
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, lens: torch.Tensor = None): # Added lens
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model's maximum sequence length ({self.max_seq_len})"
            )

        # Token embeddings
        tok_emb = self.token_embeddings(input_ids) # (B, T, D)

        # Positional embeddings
        # Create position IDs: (T) -> (1, T)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embeddings(pos) # (1, T, D)

        # Add token and positional embeddings (broadcasts pos_emb if B > 1)
        x = tok_emb + pos_emb # (B, T, D)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, lens=lens) # Pass lens to each TransformerBlock

        x = self.final_norm(x) # Final normalization
        logits = self.lm_head(x) # Language modeling head

        return logits
