import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler and more efficient alternative to LayerNorm that only scales
    the input by the root mean square of its elements. Used in Gemma, Llama, and Mistral.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Block(nn.Module):
    """
    Jommarn-Tiny Transformer block.

    Consists of RMSNorm, Hybrid Multi-Head Attention, another RMSNorm, and SwiGLU MLP.
    Includes Per-Layer Embeddings (PLE) to enhance "Intelligence Density".

    Args:
        n_head (int): Number of attention heads.
        n_embed (int): Embedding dimensionality.
        context_length (int): Maximum sequence length.
        layer_id (int): The index of this layer in the stack.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, layer_id: int, n_kv_head: int = 2) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length, n_kv_head=n_kv_head)
        self.ln2 = RMSNorm(n_embed)
        self.mlp = MLP(n_embed)
        
        # PLE: Per-Layer Embedding (learnable bias unique to this layer)
        self.ple_embedding = nn.Parameter(torch.randn(1, 1, n_embed) * 0.02)
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, is_local: bool = False, rope_cos: torch.Tensor = None, rope_sin: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor.
            is_local (bool): Whether to use Sliding Window (Local) attention.
            rope_cos (torch.Tensor): RoPE cosine frequencies.
            rope_sin (torch.Tensor): RoPE sine frequencies.
        """
        # PLE: Inject layer-specific position/depth info
        x = x + self.ple_embedding
        
        # Hybrid Attention (Local or Global)
        x = x + self.attn(self.ln1(x), is_local=is_local, rope_cos=rope_cos, rope_sin=rope_sin)
        
        # SwiGLU MLP
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor, is_local: bool = False, rope_cos: torch.Tensor = None, rope_sin: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maintained for internal feature access during training experiments.
        """
        res = x + self.ple_embedding
        res = res + self.attn(self.ln1(res), is_local=is_local, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res

if __name__ == '__main__':
    # Example Usage
    batch_size = 2
    sequence_length = 5
    embedding_dim = 768
    num_heads = 12
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    transformer_block = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, layer_id=0, n_kv_head=2)
    output_tensor = transformer_block(input_tensor)

    print("Jommarn Block Input Shape:", input_tensor.shape)
    print("Jommarn Block Output Shape:", output_tensor.shape)