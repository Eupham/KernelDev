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

# Mock SPECIAL_TOKENS for testing without data_builder dependency
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5
}

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
    
    def forward(self, x, in_span=None, span_id=None, is_prefix=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        is_causal = self.causal

        # Use flash attention kernel with metadata only (no attention_mask)
        out = flash_attention(
            q=q,
            k=k,
            v=v,
            lens=None,
            causal=is_causal,
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
    """Transformer block with pre-normalization and per-task layer uncertainty."""
    
    def __init__(self, dim, n_heads, mlp_ratio=4, causal=True, vocab_size=None, has_layer_supervision=False, task_names=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=causal)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        
        # Layer uncertainty and supervision components
        self.has_layer_supervision = has_layer_supervision
        self.task_names = task_names or []
        
        # Per-task layer uncertainty parameters - ALL layers get these now
        if task_names:
            self.log_sigmas = nn.ParameterDict()
            for task in task_names:
                # Add small random perturbation to break symmetry between layers and tasks
                init_value = torch.normal(0.0, 0.05, (1,))
                self.log_sigmas[task] = nn.Parameter(init_value)
        
        # Readout head for deep supervision (if enabled)
        if has_layer_supervision and vocab_size is not None:
            self.layer_head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, x, in_span=None, span_id=None, is_prefix=None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), in_span=in_span, span_id=span_id, is_prefix=is_prefix)
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x


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
        task_names: list = None,
        layer_supervision_frequency: int = 4  # Apply layer supervision every N layers
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.bidirectional_prefix_len = bidirectional_prefix_len
        self.vocab_size = vocab_size
        self.layer_supervision_frequency = layer_supervision_frequency
        self.task_names = task_names or []

        # Token and position embeddings (no bias)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Transformer blocks with per-task layer uncertainty for ALL layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                causal=causal,
                vocab_size=vocab_size,
                has_layer_supervision=(i % layer_supervision_frequency == 0 and i > 0),  # Keep supervision for specific layers
                task_names=task_names  # All layers get per-task uncertainty
            )
            for i in range(n_layers)
        ])
        
        # Track which layers have supervision for easier access
        self.supervised_layer_indices = [i for i in range(n_layers) 
                                       if i % layer_supervision_frequency == 0 and i > 0]
        
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
    
    def forward(self, x, targets=None, attention_mask=None, task_name=None, task_type=None, spans=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None, in_span=None, span_id=None, is_prefix=None):
        batch_size, seq_len = x.shape
        
        # Handle both task_name and task_type parameters
        if task_type is not None:
            task_name = task_type
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # Generate metadata tensors based on input parameters or tokens
        if in_span is not None and span_id is not None and is_prefix is not None:
            # Use directly provided metadata tensors
            pass  # in_span, span_id, is_prefix are already set
        elif attention_mask is not None and isinstance(attention_mask, dict):
            # Use metadata from data builder (remove task_name dependency)
            in_span = attention_mask['in_span']
            span_id = attention_mask['span_id'] 
            is_prefix = attention_mask['is_prefix']
        else:
            # Generate metadata from tokens (auto-detect sequence type)
            span_start_id = SPECIAL_TOKENS['[SPAN]']
            span_end_id = SPECIAL_TOKENS['[ES]']
            cls_token_id = SPECIAL_TOKENS['[CLS]']
            maskq_token_id = SPECIAL_TOKENS['[MASKQ]']

            # Initialize metadata tensors
            in_span = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            span_id = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)
            is_prefix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            
            # Check if this looks like a cocktail party sequence (has span tokens)
            has_spans = (x == span_start_id).any() or (x == span_end_id).any() or (x == maskq_token_id).any()
            
            if has_spans:
                # Process cocktail party style sequence
                in_span = (torch.cumsum((x == span_start_id).int(), dim=1) - torch.cumsum((x == span_end_id).int(), dim=1)) > 0
                span_id = torch.cumsum((x == span_start_id).int(), dim=1)
                span_id[~in_span] = 0  # Non-span tokens get span_id=0
                
                # Mark MASKQ tokens with special span_id=-1 (last token behavior)
                maskq_positions = (x == maskq_token_id)
                span_id[maskq_positions] = -1
                
                # Mark prefix tokens (everything up to and including [CLS])
                for batch_idx in range(batch_size):
                    cls_positions = (x[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                    if len(cls_positions) > 0:
                        cls_pos = cls_positions[0].item()
                        is_prefix[batch_idx, :cls_pos + 1] = True
            else:
                # Process teacher forcing style sequence (only [CLS] special token behavior)
                for batch_idx in range(batch_size):
                    cls_positions = (x[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
                    if len(cls_positions) > 0:
                        # Mark everything up to and including the first [CLS] as prefix
                        cls_pos = cls_positions[0].item()
                        is_prefix[batch_idx, :cls_pos + 1] = True

        # Apply transformer blocks with layer supervision
        layer_losses = {}
        intermediate_outputs = {}
        
        for i, block in enumerate(self.blocks):
            # Always use metadata tensors for attention control (no task-based routing)
            x_embed = block(x_embed, in_span=in_span, span_id=span_id, is_prefix=is_prefix)
            
            # Collect intermediate outputs for layer supervision
            if block.has_layer_supervision and targets is not None:
                # Apply normalization and compute layer logits
                normalized_x = self.norm_out(x_embed)  # Use same normalization as final layer
                layer_logits = block.layer_head(normalized_x)
                
                # Compute layer-wise cross-entropy loss
                layer_loss = F.cross_entropy(
                    layer_logits.view(-1, layer_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SPECIAL_TOKENS['[PAD]']
                )
                
                layer_losses[f'layer_{i}'] = layer_loss
                intermediate_outputs[f'layer_{i}'] = layer_logits
        
        # Final normalization
        x_embed = self.norm_out(x_embed)
        
        # Auto-detect output mode based on tokens present (no task-based routing)
        mask_token_id = SPECIAL_TOKENS['[MASK]']
        span_start_id = SPECIAL_TOKENS['[SPAN]']
        span_end_id = SPECIAL_TOKENS['[ES]']
        
        # If sequence contains span tokens and mask, process as cocktail party
        has_mask = (x == mask_token_id).any()
        has_spans = (x == span_start_id).any() and (x == span_end_id).any()
        
        if has_mask and has_spans:
            # Cocktail party processing (span-based reasoning)
            B, T = x.shape
            D = x_embed.size(-1)

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
            # Teacher forcing processing (generative language modeling)
            logits = self.head(x_embed)
            loss = None
            
            if targets is not None:
                # Compute final layer cross-entropy loss
                final_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SPECIAL_TOKENS['[PAD]']
                )
                
                # If we have layer supervision, return structured loss
                if layer_losses:
                    loss = {
                        'final_loss': final_loss,
                        'layer_losses': layer_losses
                    }
                else:
                    loss = final_loss
                    
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
