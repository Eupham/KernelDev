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
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        is_causal = self.causal and attention_mask is None

        # Use flash attention kernel
        out = flash_attention(
            q=q,
            k=k,
            v=v,
            lens=None,
            causal=is_causal,
            attention_mask=attention_mask
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
    
    def forward(self, x, attention_mask=None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
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
    
    def forward(self, x, targets=None, span_ids=None, task_name=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None):
        batch_size, seq_len = x.shape
        
        attention_mask = None
        if span_ids is not None:
            span_ids = span_ids.to(x.device)
            same_span = span_ids.unsqueeze(2) == span_ids.unsqueeze(1)
            outside_span = span_ids == 0
            attention_mask = same_span | outside_span.unsqueeze(2) | outside_span.unsqueeze(1)
            attention_mask = attention_mask.to(torch.bool)

        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x_embed = block(x_embed, attention_mask=attention_mask)
        
        x_embed = self.norm_out(x_embed)
        
        if task_name == 'cocktail_party':
            mask_token_id = SPECIAL_TOKENS['[MASK]']
            mask_positions = (x == mask_token_id).nonzero(as_tuple=True)

            if mask_positions[0].numel() == 0:
                # No mask token found, cannot proceed with this task
                return torch.empty(batch_size, 0, device=x.device), torch.tensor(0.0, device=x.device)

            batch_indices = mask_positions[0]
            sequence_indices = mask_positions[1]
            h_context = x_embed[batch_indices, sequence_indices]

            num_spans = span_ids.max()
            if num_spans == 0: # No spans found
                 return torch.empty(batch_size, 0, device=x.device), torch.tensor(0.0, device=x.device)

            span_embeddings = []

            for i in range(1, num_spans + 1):
                span_mask = (span_ids == i).unsqueeze(-1)
                masked_embeddings = x_embed * span_mask
                span_lengths = span_mask.sum(dim=1)
                summed_embeddings = masked_embeddings.sum(dim=1)
                avg_embeddings = summed_embeddings / (span_lengths + 1e-9)
                span_embeddings.append(avg_embeddings)

            h_spans = torch.stack(span_embeddings, dim=1)
            scores = (h_context.unsqueeze(1) * h_spans).sum(dim=-1)

            loss = None
            if correct_idx is not None:
                loss = F.cross_entropy(scores, correct_idx)

            return scores, loss
        elif task_name == 'soft_jigsaw':
            num_segments = span_ids.max()
            if num_segments == 0:
                return torch.empty(batch_size, 0, 0, device=x.device), torch.tensor(0.0, device=x.device)

            segment_embeddings = []

            for i in range(1, num_segments + 1):
                segment_mask = (span_ids == i).unsqueeze(-1)
                masked_embeddings = x_embed * segment_mask
                segment_lengths = segment_mask.sum(dim=1)
                summed_embeddings = masked_embeddings.sum(dim=1)
                avg_embeddings = summed_embeddings / (segment_lengths + 1e-9)
                segment_embeddings.append(avg_embeddings)

            H = torch.stack(segment_embeddings, dim=1)

            M_from_p_star = p_star.size(1)
            M_from_spans = H.size(1)

            if M_from_spans < M_from_p_star:
                padding_size = M_from_p_star - M_from_spans
                padding = torch.zeros(H.size(0), padding_size, H.size(2), device=H.device, dtype=H.dtype)
                H = torch.cat([H, padding], dim=1)
            elif M_from_spans > M_from_p_star:
                H = H[:, :M_from_p_star, :]

            S = self.permute_head(H).squeeze(-1)
            ranks, P_hat = soft_rank_to_perm(S, tau)
            loss = F.mse_loss(P_hat, p_star)
            return P_hat, loss
        elif task_name == 'distractor_loc':
            mask_logits = self.mask_head(x_embed).squeeze(-1)
            m_hat = torch.sigmoid(mask_logits)

            cls_h = x_embed[:, 0]
            ptr_pred = self.ptr_head(cls_h)
            ptr_pred = torch.sigmoid(ptr_pred)
            c_hat, l_hat = ptr_pred[:, 0], ptr_pred[:, 1]

            loss = None
            if m_star is not None and c_true is not None and l_true is not None:
                loss_mask = F.mse_loss(m_hat, m_star)
                loss_ptr = F.l1_loss(c_hat, c_true) + F.l1_loss(l_hat, l_true)
                loss = loss_mask + loss_ptr

            predictions = (m_hat, (c_hat, l_hat))
            return predictions, loss
        else:
            logits = self.head(x_embed)
            loss = None
            if targets is not None:
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
