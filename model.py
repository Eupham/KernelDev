"""
GPT Model Implementation with Hierarchical Attention Support

This module implements a GPT transformer architecture with support for both standard
causal attention and specialized hierarchical attention patterns for cocktail party tasks.

Key Components:
- GPTModel: Main transformer with configurable attention modes
- MultiHeadAttention: Attention mechanism with flash attention integration
- TransformerLayer: Standard transformer blocks with attention and feed-forward
- SwiGLU: Efficient activation function for improved performance  
- RMSNorm: RMS normalization for numerical stability

The model supports two distinct operating modes:
1. Teacher Forcing: Standard causal language modeling
2. Cocktail Party: Hierarchical attention for span-based reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from original_kernel import flash_attention

# =============================================================================
# Normalization and Activation Functions
# =============================================================================

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

# =============================================================================
# Attention Mechanism
# =============================================================================

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
    
    def forward(self, x, attention_mask=None, in_span=None, span_id=None, is_prefix=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        is_causal = self.causal

        # Use flash attention kernel
        out = flash_attention(
            q=q,
            k=k,
            v=v,
            lens=None,
            causal=is_causal,
            attention_mask=attention_mask,
            in_span=in_span,
            span_id=span_id,
            is_prefix=is_prefix
        )
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

# =============================================================================
# Transformer Components
# =============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, dim, n_heads, mlp_ratio=4, causal=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=causal)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
    
    def forward(self, x, attention_mask=None, in_span=None, span_id=None, is_prefix=None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, in_span=in_span, span_id=span_id, is_prefix=is_prefix)
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x


from data_builder import NUM_BIO_TAGS, SPECIAL_TOKENS, BIO_TAGS
from torch.distributions import Bernoulli

# =============================================================================
# Main GPT Model
# =============================================================================

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
        bidirectional_prefix_len=0,
        task_names: list = None
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.bidirectional_prefix_len = bidirectional_prefix_len
        
        # Learnable uncertainty parameters for each task
        if task_names:
            self.log_sigmas = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1)) for task in task_names
            })

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
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None, attention_mask=None, task_name=None, spans=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # Create metadata tensors
        if attention_mask is not None and isinstance(attention_mask, dict) and task_name == 'cocktail_party':
            # Use metadata from cocktail party data builder
            in_span = attention_mask['in_span']
            span_id = attention_mask['span_id']
            is_prefix = attention_mask['is_prefix']
        elif attention_mask is not None:
            # Legacy behavior: generate metadata from tokens
            span_start_id = SPECIAL_TOKENS['[SPAN]']
            span_end_id = SPECIAL_TOKENS['[ES]']
            cls_token_id = SPECIAL_TOKENS['[CLS]']

            in_span = (torch.cumsum((x == span_start_id).int(), dim=1) - torch.cumsum((x == span_end_id).int(), dim=1)) > 0
            span_id = torch.cumsum((x == span_start_id).int(), dim=1)
            span_id[~in_span] = -1
            is_prefix = (x == cls_token_id)
        else:
            # Create metadata tensors for teacher forcing
            in_span = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            span_id = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)  # Use 0 for non-span tokens
            
            # For teacher forcing, mark prefix tokens (task instructions + [CLS])
            cls_token_id = SPECIAL_TOKENS['[CLS]']
            is_prefix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            
            # Find [CLS] positions and mark everything up to and including [CLS] as prefix
            for batch_idx in range(batch_size):
                cls_positions = (x[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                if len(cls_positions) > 0:
                    # Mark everything up to and including the first [CLS] as prefix
                    cls_pos = cls_positions[0].item()
                    is_prefix[batch_idx, :cls_pos + 1] = True

        # Apply transformer blocks
        for block in self.blocks:
            if task_name == 'cocktail_party':
                # For cocktail party, don't pass the old attention_mask, use metadata tensors
                x_embed = block(x_embed, attention_mask=None, in_span=in_span, span_id=span_id, is_prefix=is_prefix)
            else:
                # For teacher forcing, use cocktail party attention but with proper prefix setup
                # This ensures prefix is bidirectional and context is causal
                x_embed = block(x_embed, attention_mask=None, in_span=in_span, span_id=span_id, is_prefix=is_prefix)
        
        # Final normalization
        x_embed = self.norm_out(x_embed)
        
        if task_name == 'cocktail_party':
            B, T = x.shape
            D = x_embed.size(-1)
            mask_token_id = SPECIAL_TOKENS['[MASK]']
            span_start_id = SPECIAL_TOKENS['[SPAN]']
            span_end_id   = SPECIAL_TOKENS['[ES]']

            # 1) Vectorized context extraction
            mask_positions = (x == mask_token_id).nonzero(as_tuple=True)
            h_context = x_embed.new_zeros(B, D)
            # Get the first mask for each batch item, if it exists
            unique_batch_idx, counts = torch.unique(mask_positions[0], return_counts=True)
            first_mask_indices = torch.cat((x.new_zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]))
            if unique_batch_idx.numel() > 0:
                 h_context[unique_batch_idx] = x_embed[unique_batch_idx, mask_positions[1][first_mask_indices]]

            # 2) Vectorized span processing
            span_starts = (x == span_start_id).nonzero()
            span_ends = (x == span_end_id).nonzero()

            if span_starts.numel() == 0:
                return torch.empty(0), torch.tensor(0.0, device=x.device)

            # Create a tensor to map each span to its batch index
            batch_indices = span_starts[:, 0]

            # Calculate max number of spans for padding
            max_spans = (x == span_start_id).sum(dim=1).max()

            h_spans = x_embed.new_zeros(B, max_spans, D)

            for i in range(B):
                st_indices = span_starts[batch_indices == i, 1]
                ed_indices = span_ends[batch_indices == i, 1]

                for j, (st, ed) in enumerate(zip(st_indices, ed_indices)):
                    if st + 1 < ed:
                        h_spans[i, j] = x_embed[i, st + 1:ed].mean(dim=0)

            # 4) Compute scores via einsum
            scores = torch.einsum('bd,bnd->bn', h_context, h_spans)

            loss = None
            if correct_idx is not None:
                loss = F.cross_entropy(scores, correct_idx)

            return scores, loss
        else:
            # Teacher forcing task (generative)
            logits = self.head(x_embed)
            loss = None
            if targets is not None:
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SPECIAL_TOKENS['[PAD]']
                )
            return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens using the model with top-k and top-p sampling."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop sequence if it gets too long
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
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
