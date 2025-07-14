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
    
    def forward(self, x, is_prefix_token_mask: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use flash attention kernel
        out = flash_attention(
            q=q,
            k=k,
            v=v,
            lens=None,
            causal=self.causal,
            is_prefix_token_mask=is_prefix_token_mask # Pass the mask
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
    
    def forward(self, x, is_prefix_token_mask: torch.Tensor | None = None):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x), is_prefix_token_mask=is_prefix_token_mask)
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
        cls_token_id: int | None = None,
        use_cls_prefix_attention: bool = True,
        n_candidates_span_selection: int = 4 # New argument for the span selection task
    ):
        super().__init__()
        self.dim = dim
        self.cls_token_id = cls_token_id
        self.use_cls_prefix_attention = use_cls_prefix_attention
        self.n_candidates_span_selection = n_candidates_span_selection # Store n_candidates
        print(f"GPTModel.__init__: cls_token_id={self.cls_token_id}, use_cls_prefix_attention={self.use_cls_prefix_attention}. Heads for NSP, Rank, and Span Selection are present.")
        self.max_seq_len = max_seq_len
        
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
        
        # NSP head for 3-class classification
        self.nsp_head = nn.Linear(self.dim, 3)  # 3 classes: 0=order, 1=out of order, 2=garbled

        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_emb.weight

        # Head for Auxiliary Task 1: Rank Regression (1 scalar per token, sigmoid output)
        self.rank_regression_head = nn.Linear(self.dim, 1)
        
        # Head for Auxiliary Task 2: Span Selection (n_candidates logits from CLS token)
        self.span_selection_head = nn.Linear(self.dim, self.n_candidates_span_selection)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None, force_disable_prefix_attention: bool = False):
        batch_size, seq_len = x.shape # x is input token IDs (batch_size, seq_len)
        
        # Initialize loss and all head outputs
        loss = None
        nsp_logits = None
        rank_regression_outputs = None
        span_selection_logits = None

        # Create current_is_prefix_token_mask based on cls_token_id and use_cls_prefix_attention
        current_is_prefix_token_mask = None
        # force_disable_prefix_attention is an argument to forward
        if self.use_cls_prefix_attention and not force_disable_prefix_attention:
            # Create a mask [True, True, False, ..., False] for the sequence length
            # This assumes Task ID (index 0) and CLS (index 1) are prefix tokens.
            # x is input_ids with shape (batch_size, seq_len)
            # seq_len is already available from x.shape
            mask_values = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
            if seq_len > 0: # Task ID token
                mask_values[0] = True
            if seq_len > 1: # CLS token
                mask_values[1] = True
            current_is_prefix_token_mask = mask_values

        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        # x_emb is (batch_size, seq_len, dim)
        x_emb = self.token_emb(x) + self.pos_emb(pos)
        
        # Apply transformer blocks
        # Pass current_is_prefix_token_mask to each block
        processed_x = x_emb
        for block in self.blocks:
            processed_x = block(processed_x, is_prefix_token_mask=current_is_prefix_token_mask)

        # Final normalization
        # processed_x is hidden states (batch_size, seq_len, dim)
        processed_x = self.norm_out(processed_x)

        # Language model logits (Primary Task)
        logits = self.head(processed_x) # (batch_size, seq_len, vocab_size)

        # Auxiliary Task 1: Rank Regression Head
        # Takes all hidden states and predicts a rank (0-1) for each token position.
        rank_regression_logits = self.rank_regression_head(processed_x) # (batch_size, seq_len, 1)
        rank_regression_outputs = torch.sigmoid(rank_regression_logits) # (batch_size, seq_len, 1)

        # Auxiliary Task (NSP-like, using CLS token representation)
        if self.cls_token_id is not None: # Indicates CLS token processing is relevant
            cls_representation = processed_x[:, 1, :] # CLS is at index 1
            nsp_logits = self.nsp_head(cls_representation)
            span_selection_logits = self.span_selection_head(cls_representation)

        if targets is not None:
            # Calculate raw per-token loss for the primary LM task, without reduction
            raw_lm_loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='none'
            )
            # Reshape per-token loss to (batch_size, seq_len)
            lm_loss_per_token_unmasked = raw_lm_loss_per_token.view(batch_size, seq_len)

            valid_targets_mask = (targets != -1).float()
            per_item_lm_loss_sum = (lm_loss_per_token_unmasked * valid_targets_mask).sum(dim=1)
            per_item_valid_token_count = valid_targets_mask.sum(dim=1)
            per_item_valid_token_count = torch.where(
                per_item_valid_token_count == 0,
                torch.ones_like(per_item_valid_token_count),
                per_item_valid_token_count
            )
            lm_loss_per_item = per_item_lm_loss_sum / per_item_valid_token_count
            loss = lm_loss_per_item
        else:
            loss = None # Loss is None if targets are not provided (e.g., during inference)

        # Return all head outputs in a dictionary for clarity and scalability
        return {
            'lm_logits': logits,
            'lm_loss': loss,
            'nsp_logits': nsp_logits,
            'rank_outputs': rank_regression_outputs,
            'span_selection_logits': span_selection_logits
        }
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_prefix_attention_in_prompt: bool = False):
        """
        Generate new tokens using the model with top-k and top-p sampling.

        Args:
            idx (torch.Tensor): Input sequence of token IDs (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float, optional): Sampling temperature. Higher values make output more random. Defaults to 1.0.
            top_k (int, optional): If set, only sample from the top k most likely next tokens. Defaults to None.
            top_p (float, optional): If set, sample from the smallest set of tokens whose cumulative probability exceeds top_p (nucleus sampling). Defaults to None.
            use_prefix_attention_in_prompt (bool, optional): If True and the model is configured for NSP
                with CLS prefix attention (`self.nsp_task=True` and `self.use_cls_prefix_attention=True`),
                CLS tokens in the input `idx` (prompt) will use their special prefix attention mechanism.
                Defaults to False, meaning CLS tokens in the prompt are treated with standard causal attention
                during this generation call, regardless of model's training configuration for NSP prefix attention.

        Returns:
            torch.Tensor: Output sequence of token IDs (batch_size, seq_len + generated_tokens).
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop sequence if it gets too long
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Forward pass now returns a dictionary. We only need 'lm_logits' for generation.
                model_outputs = self(idx_cond, force_disable_prefix_attention=(not use_prefix_attention_in_prompt))
                logits = model_outputs['lm_logits'][:, -1, :] / temperature
                
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
