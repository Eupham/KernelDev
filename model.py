import torch
import torch.nn as nn
import torch.nn.functional as F
from original_kernel import flash_attention


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
    
    def forward(self, x, roles=None):
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
            causal=is_causal,
            attention_mask=None, # Not used
            in_span=roles.get('in_span') if roles else None,
            span_id=roles.get('span_id') if roles else None,
            is_prefix=roles.get('is_prefix') if roles else None,
            is_maskq=roles.get('is_maskq') if roles else None,
            is_maskmarker=roles.get('is_mask_marker') if roles else None,
        )
        
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
    
    def forward(self, x, roles=None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), roles=roles)
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x


from data_builder import NUM_BIO_TAGS, SPECIAL_TOKENS, BIO_TAGS
from torch.distributions import Bernoulli

def soft_rank_to_perm(scores: torch.Tensor, tau: float):
    """
    scores:    [B, n]
    tau:       temperature > 0
    returns:
      ranks:    [B, n]   soft ranks ∈ [1..n]
      P_hat:    [B, n, n] row-stochastic permutation
    """
    B, n = scores.shape

    # 1) soft pairwise wins: P_ij = σ((s_j - s_i) / τ)
    diff = scores.unsqueeze(-1) - scores.unsqueeze(1)          # [B, n, n]
    P_pair = torch.sigmoid(diff / tau)                         # [B, n, n]

    # 2) soft rank = 1 + sum_j P_ij
    ranks = 1 + P_pair.sum(dim=-1)                             # [B, n]

    # 3) now convert ranks → a soft permutation matrix by
    #    comparing each pair of ranks exactly as above
    diff_r = ranks.unsqueeze(-1) - ranks.unsqueeze(1)          # [B, n, n]
    P_hat  = torch.softmax(-diff_r / tau, dim=-1)              # [B, n, n]

    return ranks, P_hat

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

        # Head for soft jigsaw task
        self.permute_head = nn.Linear(dim, 1, bias=True)

        # Heads for distractor localization task
        self.mask_head = nn.Linear(dim, 1, bias=True)
        self.ptr_head = nn.Linear(dim, 2, bias=True)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None, roles=None, task_name=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # When roles are not needed, don't allocate anything — kernels skip role logic.

        # Apply transformer blocks
        # For soft_jigsaw, we want plain attention, but the head needs roles for pooling.
        # So, we pass roles=None to the blocks but use the original roles dict in the head.
        if task_name == 'soft_jigsaw':
            for block in self.blocks:
                x_embed = block(x_embed, roles=None)
        else:
            for block in self.blocks:
                x_embed = block(x_embed, roles=roles)
        
        # Final normalization
        x_embed = self.norm_out(x_embed)
        
        if task_name == 'cocktail_party':
            B, T, D = x_embed.shape

            # Query is the embedding of [MASKQ]
            maskq_pos = (x == SPECIAL_TOKENS['[MASKQ]']).nonzero(as_tuple=False)
            if maskq_pos.numel() == 0: # Should not happen with new data builder
                return torch.empty(0), torch.tensor(0.0, device=x.device)

            # Assume one [MASKQ] per batch item
            h_q = x_embed[torch.arange(B, device=x.device), maskq_pos[:, 1]]

            # Pool span embeddings using span_id from roles
            span_ids = roles['span_id']
            max_spans = span_ids.max().item()
            if max_spans <= 0:
                 return torch.empty(0), torch.tensor(0.0, device=x.device)

            # Sanitize indices for scatter_add_: replace -1 with 0 (dummy index)
            scatter_indices = span_ids.clone()
            scatter_indices[span_ids < 0] = 0

            h_spans = x_embed.new_zeros(B, max_spans, D)
            # Create a mask for valid spans (span_id > 0)
            valid_span_mask = span_ids > 0

            # Use scatter_add_ to sum embeddings for each span
            # Need to expand span_ids to match embedding dimension
            span_ids_expanded = scatter_indices.unsqueeze(-1).expand_as(x_embed)
            # We need to offset span_ids for scatter_add since it's 0-indexed
            # and our span_ids are 1-indexed.
            # We will use a tensor of size (B, max_spans+1, D) and ignore index 0
            h_spans_sum = x_embed.new_zeros(B, max_spans + 1, D)
            h_spans_sum.scatter_add_(1, span_ids_expanded.long(), x_embed * valid_span_mask.unsqueeze(-1))

            # Count tokens in each span
            span_counts = x_embed.new_zeros(B, max_spans + 1).scatter_add_(
                1, scatter_indices.long(), valid_span_mask.float()
            )

            # Compute mean, avoiding division by zero
            h_spans_mean = h_spans_sum[:, 1:] / span_counts[:, 1:].unsqueeze(-1).clamp(min=1)

            scores = torch.einsum('bd,bnd->bn', h_q, h_spans_mean)

            loss = None
            if correct_idx is not None:
                loss = F.cross_entropy(scores, correct_idx)

            return scores, loss
        elif task_name == 'soft_jigsaw':
            B, T, D = x_embed.shape
            span_ids = roles['span_id']
            M = p_star.size(1) # Number of segments

            # Pool embeddings for each span using span_id
            max_spans_found = span_ids.max().item()
            if max_spans_found <= 0:
                # No spans found, cannot proceed
                return torch.zeros_like(p_star), torch.tensor(0.0, device=x.device)

            # Sanitize indices for scatter_add_: replace -1 with 0 (dummy index)
            scatter_indices = span_ids.clone()
            scatter_indices[span_ids < 0] = 0

            h_spans = x_embed.new_zeros(B, max_spans_found + 1, D)
            valid_span_mask = span_ids > 0
            span_ids_expanded = scatter_indices.unsqueeze(-1).expand_as(x_embed)

            h_spans.scatter_add_(1, span_ids_expanded.long(), x_embed * valid_span_mask.unsqueeze(-1))

            span_counts = x_embed.new_zeros(B, max_spans_found + 1).scatter_add_(
                1, scatter_indices.long(), valid_span_mask.float()
            )

            H_pooled = h_spans[:, 1:] / span_counts[:, 1:].unsqueeze(-1).clamp(min=1)

            # Pad or truncate to match M
            num_pooled = H_pooled.size(1)
            if num_pooled < M:
                padding = x_embed.new_zeros(B, M - num_pooled, D)
                H = torch.cat([H_pooled, padding], dim=1)
            else:
                H = H_pooled[:, :M, :]

            S = self.permute_head(H).squeeze(-1)
            ranks, P_hat = soft_rank_to_perm(S, tau)
            loss = F.mse_loss(P_hat, p_star)
            return P_hat, loss
        elif task_name == 'distractor_loc':
            # Mask head
            mask_logits = self.mask_head(x_embed).squeeze(-1)  # [B, T]
            m_hat = torch.sigmoid(mask_logits)

            # Pointer head
            cls_h = x_embed[:, 0]
            ptr_pred = self.ptr_head(cls_h)  # [B, 2]
            ptr_pred = torch.sigmoid(ptr_pred)
            c_hat, l_hat = ptr_pred[:, 0], ptr_pred[:, 1]

            loss = None
            if m_star is not None and c_true is not None and l_true is not None:
                loss_mask = F.mse_loss(m_hat, m_star)
                loss_ptr = F.mse_loss(c_hat, c_true) + F.mse_loss(l_hat, l_true)
                loss = loss_mask + loss_ptr

            predictions = (m_hat, (c_hat, l_hat))
            return predictions, loss
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
