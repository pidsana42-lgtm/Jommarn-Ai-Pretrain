import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MLP(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) MLP as used in Gemma and Llama.
    
    This replaces the standard ReLU MLP with a more expressive gated architecture.
    It expands the input, applies SiLU (Swish) to one branch, gates it with another,
    and then projects back.

    Args:
        n_embed (int): The dimensionality of the input embedding.
    """
    def __init__(self, n_embed: int) -> None:
        """
        Initializes the SwiGLU MLP module.

        Args:
            n_embed (int): The dimensionality of the input embedding.
        """
        super().__init__()
        # SwiGLU typically uses three linear transformations
        self.w1 = nn.Linear(n_embed, 4 * n_embed, bias=False) # Gate branch
        self.w2 = nn.Linear(n_embed, 4 * n_embed, bias=False) # Value branch
        self.w3 = nn.Linear(4 * n_embed, n_embed, bias=False) # Projection back

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SwiGLU MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # SwiGLU(x) = (SiLU(xW1) * xW2)W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        Partial forward pass for internal feature extraction (SiLU(xW1) * xW2).
        Maintained for compatibility with some architectural experiments.
        """
        return F.silu(self.w1(x)) * self.w2(x)

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        Final projection step.
        """
        return self.w3(x)

if __name__ == '__main__':
    # Example Usage
    batch_size = 2
    sequence_length = 3
    embedding_dim = 16
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    mlp_module = MLP(n_embed=embedding_dim)
    output_tensor = mlp_module(input_tensor)

    print("SwiGLU MLP Input Shape:", input_tensor.shape)
    print("SwiGLU MLP Output Shape:", output_tensor.shape)