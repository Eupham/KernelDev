import torch
import torch.nn as nn
import torch.nn.functional as F
from original_kernel import flash_attention as original_flash_attention_wrapper
from simple_kernel import flash_attention_func as simple_flash_attention_wrapper
import os
from typing import Optional # Add Optional

class RMSNorm(nn.Module):
    """RMS normalization without learnable weight parameters."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation function for feed-forward network."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Multi-head attention using flash attention kernel."""
    
    def __init__(self, dim, n_heads, head_dim=None, causal=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim or dim // n_heads
        self.causal = causal
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use flash attention kernel
        use_simple_kernel = os.environ.get('USE_SIMPLE_KERNEL', '0') == '1'
        if use_simple_kernel:
            # Ensure q, k, v are in the expected shape (batch, heads, seq_len, head_dim)
            # simple_flash_attention_wrapper expects (q, k, v, causal, sm_scale)
            # Current q,k,v are (B, H, T, D)
            out = simple_flash_attention_wrapper(q, k, v, causal=self.causal, sm_scale=None)
            # print("Using simple_kernel.flash_attention_func (CPU fallback or alternative)") # For debugging
        else:
            out = original_flash_attention_wrapper(
                q=q,
                k=k,
                v=v,
                lens=None, # Original code passes None here
                causal=self.causal,
                sm_scale=None # Rely on default scaling in original_kernel
            )
            # print("Using original_kernel.flash_attention") # For debugging
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, dim, n_heads, mlp_ratio=4, causal=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=causal)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
    
    def forward(self, x):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """GPT-styled model using flash attention kernel."""
    
    def __init__(
        self,
        vocab_size,
        dim=768,
        n_layers=12,
        n_heads=12,
        max_seq_len=2048,
        mlp_ratio=4,
        causal=True,
        num_nsp_labels=2,
        enable_word_order_task: bool = False # Added WOD config
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.enable_word_order_task = enable_word_order_task
        
        # Token and position embeddings (no bias)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                causal=causal
            )
            for _ in range(n_layers)
        ])
        
        # Final norm and output projection
        self.norm_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False) # LM head

        # NSP head (conditional initialization or always init if config drives its usage)
        # For simplicity in forward, let's assume it's always initialized if num_nsp_labels > 0,
        # but its output is only used if nsp_labels are provided.
        # Or, make it None if not used (requires checks in forward).
        # The current structure has it initialized based on num_nsp_labels.
        self.nsp_head = nn.Linear(dim, num_nsp_labels) if num_nsp_labels > 0 else None

        # WOD head
        if self.enable_word_order_task:
            self.word_order_head = nn.Linear(self.dim, 1) # Regression to a score (0-1)
        else:
            self.word_order_head = None
        
        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None, nsp_labels=None, word_order_score_targets: Optional[torch.Tensor] = None): # Renamed wod_labels
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        x = self.token_emb(x) + self.pos_emb(pos)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm_out(x)
        
        # LM logits
        lm_logits = self.head(x)

        # NSP logits & WOD logits (both can use the first token's hidden state)
        pooled_output = x[:, 0]

        nsp_logits = None
        if self.nsp_head is not None:
            nsp_logits = self.nsp_head(pooled_output)

        predicted_word_order_score = None # Changed from word_order_logits
        if self.word_order_head is not None:
            word_order_logit = self.word_order_head(pooled_output) # Raw output
            predicted_word_order_score = torch.sigmoid(word_order_logit) # Apply sigmoid

        lm_loss, nsp_loss, word_order_loss = None, None, None

        if targets is not None:
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        if nsp_logits is not None and nsp_labels is not None and self.nsp_head is not None:
            nsp_loss = F.cross_entropy(
                nsp_logits.view(-1, self.nsp_head.out_features),
                nsp_labels.view(-1)
            )

        if predicted_word_order_score is not None and word_order_score_targets is not None and self.word_order_head is not None:
            # Ensure targets are float for MSE loss
            word_order_score_targets = word_order_score_targets.float()
            # Squeeze predicted score if it has a trailing dim of 1, to match target shape (batch_size)
            word_order_loss = F.mse_loss(predicted_word_order_score.squeeze(-1), word_order_score_targets)

        return lm_logits, nsp_logits, predicted_word_order_score, lm_loss, nsp_loss, word_order_loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens using the model with top-k and top-p sampling."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop sequence if it gets too long
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Forward pass
                # self(idx_cond) now returns: lm_logits, nsp_logits, word_order_logits, lm_loss, nsp_loss, word_order_loss
                # We only need lm_logits for generation.
                lm_logits, _, _, _, _, _ = self(idx_cond)
                logits = lm_logits[:, -1, :] / temperature # Use lm_logits
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p (nucleus) sampling if specified
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for tokens to remove
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
