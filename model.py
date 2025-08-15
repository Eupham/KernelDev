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


from typing import Optional, Dict

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional routing to PyTorch's SDPA or a custom kernel."""
    
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
    
    def forward(self, x, roles: Optional[Dict[str, torch.Tensor]] = None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if roles is None:
            # Route to PyTorch's scaled_dot_product_attention for plain causal tasks
            # This path is taken for teacher forcing and distractor localization
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        else:
            # Route to custom kernel for role-based attention
            # The kernel will handle the complex masking logic based on the `roles` tensors
            out = flash_attention(
                q=q, k=k, v=v,
                roles=roles,
                causal=self.causal
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
    
    def forward(self, x, roles: Optional[Dict[str, torch.Tensor]] = None):
        # Pre-norm for attention, passing roles through
        x = x + self.attn(self.norm1(x), roles=roles)
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x


from data_builder import SPECIAL_TOKENS
from torch.distributions import Bernoulli

def soft_rank_to_perm(scores: torch.Tensor, tau: float):
    B, n = scores.shape
    diff = scores.unsqueeze(-1) - scores.unsqueeze(1)
    P_pair = torch.sigmoid(diff / tau)
    ranks = 1 + P_pair.sum(dim=-1)
    diff_r = ranks.unsqueeze(-1) - ranks.unsqueeze(1)
    P_hat  = torch.softmax(-diff_r / tau, dim=-1)
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
        self.causal = causal
        self.bidirectional_prefix_len = bidirectional_prefix_len
        
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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, roles=None, task_name=None, targets=None, correct_idx=None, p_star=None, tau=0.1, m_star=None, c_true=None, l_true=None):
        batch_size, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x_embed = block(x_embed, roles=roles)
        
        x_embed = self.norm_out(x_embed)
        
        if task_name in ['cocktail_party', 'soft_jigsaw']:
            span_id = roles['span_id']
            assert span_id is not None, "span_id must be provided for span-based tasks"
            # Sanity check: Ensure there are spans to pool. Fail fast if not.
            assert span_id.max() >= 1, f"Task '{task_name}' requires at least one span, but none were found in `roles['span_id']`."

            safe_span_id = span_id.clamp_min(0)
            num_spans = safe_span_id.max().item()

            one_hot_ids = F.one_hot(safe_span_id, num_classes=num_spans + 1).float()
            summed_spans = torch.einsum('btd,btm->bmd', x_embed, one_hot_ids)
            span_counts = one_hot_ids.sum(dim=1).clamp_min(1e-6)
            mean_pooled_spans = summed_spans / span_counts.unsqueeze(-1)
            h_spans = mean_pooled_spans[:, 1:, :]

            if task_name == 'cocktail_party':
                is_maskq = roles['is_maskq']
                assert is_maskq.sum().item() == batch_size, "Each sequence in batch must have exactly one [MASKQ]"

                maskq_indices = is_maskq.nonzero(as_tuple=True)
                h_q = x_embed[maskq_indices[0], maskq_indices[1]]

                scores = torch.einsum('bd,bmd->bm', h_q, h_spans)

                loss = F.cross_entropy(scores, correct_idx) if correct_idx is not None else None
                return scores, loss

            elif task_name == 'soft_jigsaw':
                S = self.permute_head(h_spans).squeeze(-1)
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
            if m_star is not None:
                loss_mask = F.mse_loss(m_hat, m_star)
                loss_ptr = F.mse_loss(c_hat, c_true) + F.mse_loss(l_hat, l_true)
                loss = loss_mask + loss_ptr
            return (m_hat, (c_hat, l_hat)), loss

        else: # 'teacher_forcing'
            logits = self.head(x_embed)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SPECIAL_TOKENS.get('[PAD]', 0)
                )
            return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                logits, _ = self(idx_cond)
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
