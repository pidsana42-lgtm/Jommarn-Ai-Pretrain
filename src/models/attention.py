import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Partial Rotary Positional Embedding (p-RoPE) to a tensor.
    Rotates only the first half of the dimensions in each head.
    Supports GQA shapes.
    """
    B, T, H, D = x.shape
    d_rope = D // 2 # Partial RoPE: rotate only half
    x_rope = x[..., :d_rope]
    x_pass = x[..., d_rope:]

    # RoPE rotation: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    x_rope_left = x_rope[..., 0::2]
    x_rope_right = x_rope[..., 1::2]
    
    # [B, T, H, d_rope//2] * [1, T, 1, d_rope//2]
    cos = cos[:T, :d_rope//2].unsqueeze(0).unsqueeze(2)
    sin = sin[:T, :d_rope//2].unsqueeze(0).unsqueeze(2)
    
    rotated_left = x_rope_left * cos - x_rope_right * sin
    rotated_right = x_rope_left * sin + x_rope_right * cos
    
    # Recombine
    x_rope = torch.stack([rotated_left, rotated_right], dim=-1).flatten(-2)
    return torch.cat([x_rope, x_pass], dim=-1)

class MultiHeadAttention(nn.Module):
    """
    Jommarn-Tiny Grouped-Query Attention (GQA) - Qwen2.5/Llama-3 Styled.
    
    Implements:
    1. GQA (Grouped-Query Attention) with n_kv_head=2 for memory optimization.
    2. Partial RoPE for enhanced spatial awareness.
    3. Hybrid Attention (Sliding Window or Global).
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, n_kv_head: int = 2) -> None:
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_size = n_embed // n_head
        self.context_length = context_length
        self.window_size = context_length // 2 # Default sliding window size
        
        self.num_queries_per_kv = n_head // n_kv_head

        # Projections: Query, Key, and Value
        # Key/Value projections are smaller because of GQA
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_kv_head * self.head_size, bias=False)
        self.v_proj = nn.Linear(n_embed, n_kv_head * self.head_size, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        
        # Causal mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats Key/Value heads to match Query heads for attention computation"""
        if n_rep == 1:
            return x
        B, T, H, D = x.shape
        # Expand heads dimension: (B, T, H, 1, D) -> (B, T, H, n_rep, D) -> flatten to (B, T, H * n_rep, D)
        return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)

    def forward(self, x: torch.Tensor, is_local: bool = False, rope_cos: torch.Tensor = None, rope_sin: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Projections
        q = self.q_proj(x) # (B, T, n_head * head_size)
        k = self.k_proj(x) # (B, T, n_kv_head * head_size)
        v = self.v_proj(x) # (B, T, n_kv_head * head_size)
        
        # Reshape for multi-head/GQA: (B, T, H, D)
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_kv_head, self.head_size)
        v = v.view(B, T, self.n_kv_head, self.head_size)
        
        # Apply p-RoPE (applied before repeat_kv)
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            
        # Repeat Keys and Values to match Query head count
        k = self.repeat_kv(k, self.num_queries_per_kv) # (B, T, n_head, head_size)
        v = self.repeat_kv(v, self.num_queries_per_kv) # (B, T, n_head, head_size)
        
        # Transpose for attention to match SDPA input requirements: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ✅ Modern SDPA Optimization (Auto-selects optimal backend for CUDA / ROCm)
        if is_local:
            # Custom causal mask with sliding window
            mask = self.tril[:T, :T]
            local_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=-self.window_size)
            attn_mask = (mask * local_mask).bool().unsqueeze(0).unsqueeze(1) # shape (1, 1, T, T)
            
            # Run SDPA with custom local mask
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            # Global Causal Attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        
        # Reshape back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

if __name__ == '__main__':
    # Example Usage (GQA: 12 Query heads, 2 KV heads, n_embed 768)
    B, T, C = 2, 8, 768
    H = 12
    mha = MultiHeadAttention(n_head=H, n_embed=C, context_length=T, n_kv_head=2)
    x = torch.randn(B, T, C)
    
    # Mock RoPE
    cos = torch.cos(torch.randn(T, (C//H)//4))
    sin = torch.sin(torch.randn(T, (C//H)//4))
    
    out = mha(x, is_local=True, rope_cos=cos, rope_sin=sin)
    print("GQA MHA Output Shape:", out.shape)