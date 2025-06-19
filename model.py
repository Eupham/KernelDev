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
        nsp_task: bool = False,
        use_cls_prefix_attention: bool = True, # New parameter
    ):
        super().__init__()
        self.dim = dim
        self.cls_token_id = cls_token_id
        self.nsp_task = nsp_task
        self.use_cls_prefix_attention = use_cls_prefix_attention # Store it
        # Updated print to include the new parameter
        print(f"GPTModel.__init__: nsp_task={self.nsp_task}, cls_token_id={self.cls_token_id}, use_cls_prefix_attention={self.use_cls_prefix_attention}")
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
        
        # NSP head
        self.nsp_head = nn.Linear(dim, 1) # Binary classification for NSP

        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # Initialize bias for NSP head if it exists
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None, force_disable_prefix_attention: bool = False):
        batch_size, seq_len = x.shape # x is input token IDs (batch_size, seq_len)
        
        # Initialize loss and nsp_logits
        loss = None
        nsp_logits = None

        # Create current_is_prefix_token_mask based on cls_token_id and use_cls_prefix_attention
        current_is_prefix_token_mask = None
        # Only create prefix mask if NSP task is on, CLS token is defined, specific prefix attention for CLS is enabled, AND not forcibly disabled
        if self.nsp_task and \
           self.use_cls_prefix_attention and \
           self.cls_token_id is not None and \
           not force_disable_prefix_attention: # Check the new flag
            # The mask is (seq_len,) indicating True for CLS token positions based on the first batch item.
            # This is because original_kernel.flash_attention expects a (T,) mask.
            prefix_mask_bool_first_item = (x[0] == self.cls_token_id)
            if prefix_mask_bool_first_item.any():
                current_is_prefix_token_mask = prefix_mask_bool_first_item.to(x.device)

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

        # Language model logits
        logits = self.head(processed_x) # (batch_size, seq_len, vocab_size)

        # NSP logits calculation
        # processed_x is the output of self.norm_out(x), shape (batch_size, seq_len, dim)
        # nsp_logits = None # Already initialized at the beginning of the method
        if self.nsp_task:
            # Assuming NSPDataset places the CLS token at index 0
            # And its representation is taken from the output hidden states.
            cls_token_representation = processed_x[:, 0, :]
            nsp_logits = self.nsp_head(cls_token_representation)

        if targets is not None:
            # Compute cross-entropy loss for language modeling
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1 # Typically -100 for ignored tokens, but -1 if vocab doesn't collide
            )

        return logits, loss, nsp_logits
    
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
                
                # Forward pass - update to expect three return values
                # Pass force_disable_prefix_attention based on not use_prefix_attention_in_prompt
                logits, _, _ = self(idx_cond, force_disable_prefix_attention=(not use_prefix_attention_in_prompt))
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
