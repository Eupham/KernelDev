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
        layer_supervision_frequency: int = 4,  # Apply layer supervision every N layers
        supervise_layers: str = "frequency",  # "all" or "frequency"
        share_heads: bool = False,  # Use shared head across layers
        conditioning: str = "film"  # "film" or "concat2"
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.bidirectional_prefix_len = bidirectional_prefix_len
        self.vocab_size = vocab_size
        self.layer_supervision_frequency = layer_supervision_frequency
        self.task_names = task_names or []
        self.supervise_layers = supervise_layers
        self.share_heads = share_heads
        self.conditioning = conditioning

        # Token and position embeddings (no bias)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Determine layer supervision logic
        if supervise_layers == "all":
            layer_has_supervision = lambda i: True  # All layers have supervision
        else:
            layer_has_supervision = lambda i: (i % layer_supervision_frequency == 0 and i > 0)
        
        # Transformer blocks with per-task layer uncertainty for ALL layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                causal=causal,
                vocab_size=vocab_size if not share_heads else None,  # No individual heads if sharing
                has_layer_supervision=layer_has_supervision(i),
                task_names=task_names  # All layers get per-task uncertainty
            )
            for i in range(n_layers)
        ])
        
        # Track which layers have supervision for easier access
        if supervise_layers == "all":
            self.supervised_layer_indices = list(range(n_layers))
        else:
            self.supervised_layer_indices = [i for i in range(n_layers) 
                                           if i % layer_supervision_frequency == 0 and i > 0]
        
        # Shared head configuration
        if share_heads:
            # Shared teacher forcing head for all layers
            conditioning_dim = 2 if conditioning == "concat2" else 0
            self.shared_tf_head = nn.Linear(dim + conditioning_dim, vocab_size, bias=False)
            
            # FiLM conditioning parameters
            if conditioning == "film":
                self.layer_alpha = nn.Parameter(torch.zeros(n_layers))
                self.layer_beta = nn.Parameter(torch.zeros(n_layers))
                self.task_alpha = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in self.task_names})
                self.task_beta = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in self.task_names})
            elif conditioning == "concat2":
                self.layer_id = nn.Parameter(torch.zeros(n_layers, 1))  # scalar per layer
                self.task_id = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in self.task_names})
        
        # Final norm and output projection
        self.norm_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_emb.weight
        if share_heads:
            # Also tie shared head weights
            self.shared_tf_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def condition_hidden(self, h, layer_idx, task):
        """Apply FiLM-style conditioning: h' = (1 + α_ℓ + α_t) * h + (β_ℓ + β_t)"""
        if self.conditioning != "film":
            raise ValueError("condition_hidden only works with FiLM conditioning")
        
        a = 1.0 + self.layer_alpha[layer_idx] + self.task_alpha[task]
        b = self.layer_beta[layer_idx] + self.task_beta[task]
        return h * a + b
    
    def augment_hidden(self, h, layer_idx, task):
        """Concatenate [s_layerID, s_taskID] to each token hidden before the shared LM head"""
        if self.conditioning != "concat2":
            raise ValueError("augment_hidden only works with concat2 conditioning")
        
        B, T, D = h.shape
        lid = self.layer_id[layer_idx].expand(B, T, 1)
        tid = self.task_id[task].expand(B, T, 1)
        return torch.cat([h, lid, tid], dim=-1)  # -> (B,T,D+2)
    
    def extract_mask_query(self, h, tokens):
        """Extract context vector from mask query tokens for cocktail party task"""
        mask_token_id = SPECIAL_TOKENS['[MASK]']
        B, T, D = h.shape
        
        # Vectorized context extraction
        mask_positions = (tokens == mask_token_id).nonzero(as_tuple=True)
        h_context = h.new_zeros(B, D)
        
        # Get the first mask for each batch item, if it exists
        if mask_positions[0].numel() > 0:
            unique_batch_idx, counts = torch.unique(mask_positions[0], return_counts=True)
            first_mask_indices = torch.cat((tokens.new_zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]))
            if unique_batch_idx.numel() > 0:
                h_context[unique_batch_idx] = h[unique_batch_idx, mask_positions[1][first_mask_indices]]
        
        return h_context
    
    def pool_spans(self, h, tokens):
        """Pool span embeddings for cocktail party task"""
        span_start_id = SPECIAL_TOKENS['[SPAN]']
        span_end_id = SPECIAL_TOKENS['[ES]']
        B, T, D = h.shape
        
        # Vectorized span processing
        span_starts = (tokens == span_start_id).nonzero()
        span_ends = (tokens == span_end_id).nonzero()
        
        if span_starts.numel() == 0:
            return h.new_zeros(B, 1, D)  # Return at least one span dimension
        
        # Create a tensor to map each span to its batch index
        batch_indices = span_starts[:, 0]
        
        # Calculate max number of spans for padding
        max_spans = (tokens == span_start_id).sum(dim=1).max()
        
        h_spans = h.new_zeros(B, max_spans, D)
        
        for i in range(B):
            st_indices = span_starts[batch_indices == i, 1]
            ed_indices = span_ends[batch_indices == i, 1]
            
            for j, (st, ed) in enumerate(zip(st_indices, ed_indices)):
                if st + 1 < ed:
                    h_spans[i, j] = h[i, st + 1:ed].mean(dim=0)
        
        return h_spans
    
    def final_span_scores(self, h, tokens):
        """Compute final span scores for cocktail party task (reuses existing logic)"""
        ctx_vec = self.extract_mask_query(h, tokens)  # (B, D)
        span_vecs = self.pool_spans(h, tokens)  # (B, Nspans, D)
        scores = torch.einsum('bd,bnd->bn', ctx_vec, span_vecs)
        return scores
    
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

        # Apply transformer blocks with layer supervision for BOTH tasks
        layer_losses_tf = {}  # Teacher forcing layer losses
        layer_losses_cp = {}  # Cocktail party layer losses
        
        for i, block in enumerate(self.blocks):
            # Always use metadata tensors for attention control
            x_embed = block(x_embed, in_span=in_span, span_id=span_id, is_prefix=is_prefix)
            
            # Apply normalization for layer-wise supervision
            if block.has_layer_supervision:
                h = self.norm_out(x_embed)  # Use same normalization as final layer
                
                # --- Teacher Forcing per-layer logits (shared head) ---
                if targets is not None:
                    if self.share_heads:
                        # Use shared head with conditioning
                        if self.conditioning == "film":
                            h_conditioned = self.condition_hidden(h, i, "teacher_forcing")
                            layer_logits = self.shared_tf_head(h_conditioned)
                        else:  # "concat2"
                            h_aug = self.augment_hidden(h, i, "teacher_forcing")
                            layer_logits = self.shared_tf_head(h_aug)
                    else:
                        # Use per-layer head (existing logic)
                        layer_logits = block.layer_head(h)
                    
                    # CE over tokens (mask/pad as usual)
                    ce_tf = F.cross_entropy(
                        layer_logits.view(-1, layer_logits.size(-1)),
                        targets.view(-1),
                        ignore_index=SPECIAL_TOKENS['[PAD]']
                    )
                    layer_losses_tf[f'layer_{i}'] = ce_tf
                
                # --- Cocktail Party per-layer scores (contrastive span selection) ---
                if correct_idx is not None:
                    # Check if this is actually a cocktail party sequence
                    mask_token_id = SPECIAL_TOKENS['[MASK]']
                    span_start_id = SPECIAL_TOKENS['[SPAN]']
                    span_end_id = SPECIAL_TOKENS['[ES]']
                    has_mask = (x == mask_token_id).any()
                    has_spans = (x == span_start_id).any() and (x == span_end_id).any()
                    
                    if has_mask and has_spans:
                        # Compute context and span embeddings from h for this layer
                        ctx_vec = self.extract_mask_query(h, x)  # (B, D)
                        span_vecs = self.pool_spans(h, x)  # (B, Nspans, D)
                        
                        # Scores: dot(ctx, span) per candidate
                        scores_i = torch.einsum('bd,bnd->bn', ctx_vec, span_vecs)
                        ce_cp = F.cross_entropy(scores_i, correct_idx)  # (B,)
                        layer_losses_cp[f'layer_{i}'] = ce_cp
        
        # Final normalization
        x_embed = self.norm_out(x_embed)
        
        # Auto-detect output mode based on tokens present
        mask_token_id = SPECIAL_TOKENS['[MASK]']
        span_start_id = SPECIAL_TOKENS['[SPAN]']
        span_end_id = SPECIAL_TOKENS['[ES]']
        
        # Check what type of processing we need
        has_mask = (x == mask_token_id).any()
        has_spans = (x == span_start_id).any() and (x == span_end_id).any()
        
        # Prepare final outputs
        logits_final = None
        scores_final = None
        loss_tf = None
        loss_cp = None
        
        # Teacher forcing final head
        if targets is not None:
            logits_final = self.head(x_embed)
            final_ce_tf = F.cross_entropy(
                logits_final.view(-1, logits_final.size(-1)),
                targets.view(-1),
                ignore_index=SPECIAL_TOKENS['[PAD]']
            )
            
            # Structure teacher forcing loss
            if layer_losses_tf:
                loss_tf = {
                    'final_ce': final_ce_tf,
                    'layer_ce': layer_losses_tf
                }
            else:
                loss_tf = final_ce_tf
        
        # Cocktail party final scores
        if correct_idx is not None and has_mask and has_spans:
            scores_final = self.final_span_scores(x_embed, x)
            final_ce_cp = F.cross_entropy(scores_final, correct_idx)
            
            # Structure cocktail party loss
            if layer_losses_cp:
                loss_cp = {
                    'final_ce': final_ce_cp,
                    'layer_ce': layer_losses_cp
                }
            else:
                loss_cp = final_ce_cp
        
        # Return based on what was requested
        if has_mask and has_spans:
            # Cocktail party mode - return structured loss for both tasks if available
            if loss_tf is not None and loss_cp is not None:
                # Both tasks available
                combined_loss = {"teacher_forcing": loss_tf, "cocktail_party": loss_cp}
            elif loss_cp is not None:
                # Only cocktail party
                combined_loss = loss_cp
            else:
                combined_loss = None
            
            return scores_final, combined_loss
        else:
            # Teacher forcing mode
            return logits_final, loss_tf
    
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
