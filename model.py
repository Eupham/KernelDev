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
    
    def forward(self, x, use_role_mask=False, roles=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # The flash_attention function now handles routing to SDPA or the custom kernel
        out, _ = flash_attention(
            q=q,
            k=k,
            v=v,
            causal=self.causal,
            use_role_mask=use_role_mask,
            roles=roles
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
    
    def forward(self, x, use_role_mask=False, roles=None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), use_role_mask=use_role_mask, roles=roles)
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
    """GPT-styled model using flash attention kernel with role-based masking."""
    
    def __init__(
        self,
        vocab_size,
        dim=768,
        n_layers=12,
        n_heads=12,
        max_seq_len=2048,
        mlp_ratio=4,
        causal=True,
        bidirectional_prefix_len=0, # Unused, roles handle this
        task_names: list = None
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        if task_names:
            self.log_sigmas = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1)) for task in task_names
            })

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, causal=causal)
            for _ in range(n_layers)
        ])
        
        self.norm_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.permute_head = nn.Linear(dim, 1, bias=True)
        self.mask_head = nn.Linear(dim, 1, bias=True)
        self.ptr_head = nn.Linear(dim, 2, bias=True)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _pool_spans(self, x_embed, roles):
        """Pool embeddings based on span_id using a safe scatter-mean."""
        B, T, D = x_embed.shape
        span_id = roles['span_id']

        # Ensure M is at least 1 to avoid errors with empty spans
        M = max(1, span_id.max().item())

        # Remap padding (-1) to a safe bin (0) that we will ignore
        span_ids_remapped = span_id.clone()
        span_ids_remapped[span_id == -1] = 0

        # Create buffer for sums and counts (M+1 bins, bin 0 is for non-span/padding)
        sum_buffer = torch.zeros(B, M + 1, D, device=x_embed.device, dtype=x_embed.dtype)
        count_buffer = torch.zeros(B, M + 1, 1, device=x_embed.device, dtype=x_embed.dtype)

        # Use scatter_add_ to sum embeddings and counts for each span
        idx = span_ids_remapped.unsqueeze(-1).expand_as(x_embed)
        sum_buffer.scatter_add_(1, idx, x_embed)
        count_buffer.scatter_add_(1, span_ids_remapped.unsqueeze(-1), torch.ones_like(span_ids_remapped).float().unsqueeze(-1))

        # Compute mean, avoiding division by zero
        mean_embeddings = sum_buffer / (count_buffer + 1e-9)
        
        # Return only the actual span embeddings (bins 1 to M)
        return mean_embeddings[:, 1:, :]

    def forward(self, x, targets=None, roles=None, task_name=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None):
        batch_size, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # Determine if role-based masking should be used in the transformer blocks
        use_role_mask = task_name == 'cocktail_party'

        # For soft jigsaw, roles are used only in the head, not in the attention blocks.
        # For teacher forcing and distractor loc, roles are not used at all in attention.
        transformer_roles = roles if use_role_mask else None

        for block in self.blocks:
            x_embed = block(x_embed, use_role_mask=use_role_mask, roles=transformer_roles)
        
        x_embed = self.norm_out(x_embed)
        
        if task_name == 'cocktail_party':
            assert roles is not None, "Cocktail party task requires roles."

            # Query is the embedding of the [MASKQ] token
            maskq_positions = roles['is_maskq'].nonzero(as_tuple=True)
            h_query = x_embed[maskq_positions[0], maskq_positions[1]]

            # Pool span embeddings
            h_spans = self._pool_spans(x_embed, roles) # [B, M, D]

            # Compute scores and loss
            scores = torch.einsum('bd,bmd->bm', h_query, h_spans)
            loss = F.cross_entropy(scores, correct_idx) if correct_idx is not None else None
            return scores, loss

        elif task_name == 'soft_jigsaw':
            assert roles is not None, "Soft jigsaw task requires roles."

            # Pool span embeddings to get sentence representations
            H = self._pool_spans(x_embed, roles) # [B, M, D]

            # Project to get scores for ranking
            S = self.permute_head(H).squeeze(-1)
            ranks, P_hat = soft_rank_to_perm(S, tau)
            loss = F.mse_loss(P_hat, p_star) if p_star is not None else None
            return P_hat, loss

        elif task_name == 'distractor_loc':
            mask_logits = self.mask_head(x_embed).squeeze(-1)
            m_hat = torch.sigmoid(mask_logits)

            cls_h = x_embed[:, 0]
            ptr_pred = torch.sigmoid(self.ptr_head(cls_h))
            c_hat, l_hat = ptr_pred[:, 0], ptr_pred[:, 1]

            loss = None
            if m_star is not None and c_true is not None and l_true is not None:
                loss = F.mse_loss(m_hat, m_star) + F.mse_loss(c_hat, c_true) + F.mse_loss(l_hat, l_true)

            return (m_hat, (c_hat, l_hat)), loss

        else: # Teacher forcing
            logits = self.head(x_embed)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=SPECIAL_TOKENS['[PAD]'])
            return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens using the model with top-k and top-p sampling."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Use teacher forcing path for generation
                logits, _ = self(idx_cond, task_name='teacher_forcing')
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
