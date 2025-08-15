import torch
import torch.nn as nn
import torch.nn.functional as F
from original_kernel import flash_attention
from typing import Optional, Dict

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # No learnable weight, using a static '1'
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS, add epsilon, then compute reciprocal square root
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply the learnable weight
        return norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
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
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Use PyTorch's SDPA for plain causal attention if no roles are provided
        if roles is None:
            # Transpose for SDPA: (B, T, H, D) -> (B, H, T, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
            # Transpose back: (B, H, T, D) -> (B, T, H, D)
            out = out.transpose(1, 2)
        else:
            # Use custom flash attention kernel for role-based masking
            # Kernel expects (B, H, T, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # Ensure role tensors are contiguous
            for key, tensor in roles.items():
                if tensor is not None:
                    assert tensor.is_contiguous(), f"Role tensor '{key}' is not contiguous."

            out = flash_attention(q, k, v, causal=self.causal, roles=roles)
            # Transpose back: (B, H, T, D) -> (B, T, H, D)
            out = out.transpose(1, 2)

        out = out.contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, causal=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=causal)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
    
    def forward(self, x, roles: Optional[Dict[str, torch.Tensor]] = None):
        x = x + self.attn(self.norm1(x), roles=roles)
        x = x + self.mlp(self.norm2(x))
        return x

from data_builder import SPECIAL_TOKENS

def soft_rank_to_perm(scores: torch.Tensor, tau: float):
    diff = scores.unsqueeze(-1) - scores.unsqueeze(1)
    P_pair = torch.sigmoid(diff / tau)
    ranks = 1 + P_pair.sum(dim=-1)
    diff_r = ranks.unsqueeze(-1) - ranks.unsqueeze(1)
    P_hat  = torch.softmax(-diff_r / tau, dim=-1)
    return ranks, P_hat

class GPTModel(nn.Module):
    def __init__(
        self, vocab_size, dim=768, n_layers=12, n_heads=12,
        max_seq_len=2048, mlp_ratio=4, causal=True,
        bidirectional_prefix_len=0, task_names: list = None
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.bidirectional_prefix_len = bidirectional_prefix_len
        
        if task_names:
            self.log_sigmas = nn.ParameterDict({task: nn.Parameter(torch.zeros(1)) for task in task_names})

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
    
    def forward(self, x, targets=None, roles: Optional[Dict[str, torch.Tensor]] = None, task_name=None,
                correct_idx=None, p_star=None, tau=0.5, m_star=None, c_true=None, l_true=None):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # No more dummy role creation, they are passed in or None
        for block in self.blocks:
            x_embed = block(x_embed, roles=roles)
        
        x_embed = self.norm_out(x_embed)
        
        if task_name == 'cocktail_party' or task_name == 'soft_jigsaw':
            # Sanity checks for span-based tasks
            assert roles is not None, "Roles must be provided for span-based tasks."
            assert roles['is_maskq'].sum(dim=1).allclose(torch.ones(B, device=x.device)), "Exactly one [MASKQ] per sequence required."
            assert roles['span_id'].max() >= 1, "At least one span must be present."

            # Strict span pooling using span_id
            span_id = roles['span_id']
            # Remap -1 (no span) to 0 for one-hot encoding
            span_id_remapped = span_id.clamp_min(0)

            num_spans = span_id.max().item()
            if num_spans == 0: # Handle case with no spans if it slips through assertion
                 num_spans = 1

            one_hot_ids = F.one_hot(span_id_remapped, num_classes=num_spans + 1).float()

            # Create a mask to exclude pooling [SPAN] and [ES] tokens
            is_marker = (x == SPECIAL_TOKENS['[SPAN]']) | (x == SPECIAL_TOKENS['[ES]'])
            # We only want to pool tokens that are `in_span` but are not markers
            pool_mask = (roles['in_span'] & ~is_marker).unsqueeze(-1)

            # Sum embeddings per span
            masked_one_hot = one_hot_ids * pool_mask
            summed_embs = torch.einsum('btd,btn->bnd', x_embed, masked_one_hot)

            # Count tokens per span
            span_counts = masked_one_hot.sum(dim=1).clamp_min(1e-6)

            # Compute mean pooled embeddings
            pooled_embs = summed_embs / span_counts

            # Slice away the 0-th bin (non-span tokens)
            H_spans = pooled_embs[:, 1:, :] # Shape: [B, num_spans, D]

            if task_name == 'cocktail_party':
                # Query is the embedding at the [MASKQ] position
                h_q = x_embed[roles['is_maskq']] # Shape: [B, D]

                scores = torch.einsum('bd,bnd->bn', h_q, H_spans)
                loss = F.cross_entropy(scores, correct_idx) if correct_idx is not None else None
                return scores, loss

            else: # soft_jigsaw
                S = self.permute_head(H_spans).squeeze(-1)
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

        else: # teacher_forcing
            logits = self.head(x_embed)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=SPECIAL_TOKENS['[PAD]'])
            return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # For generation, roles are None, so it will use SDPA
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
