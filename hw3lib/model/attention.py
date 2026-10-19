import torch.nn as nn
import torch
from typing import Tuple, Optional
from .norms import RMSNorm
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

This file contains the multi-head attention used by the transformer sublayers:

MultiHeadAttention: Scaled dot-product attention over multiple heads
- Mirrors the mytorch.nn.MultiHeadAttention you implemented in HW3P1
- Optionally applies rotary embeddings to the queries and keys
- Optionally applies QK-Norm to the queries and keys

Porting your HW3P1 implementation:
- Subclass nn.Module and call super().__init__()
- Replace the mytorch Linear layers with nn.Linear(embed_dim, embed_dim)
- Replace the numpy operations with their torch equivalents
- Drop backward and init_weights, neither is needed here
- Return the attention weights alongside the output

The method names and the forward signature are unchanged from HW3P1, so keep
that file open beside this one.

Specification:
- key_padding_mask has shape (N, S), attn_mask has shape (L, S), and in both
  True marks a position to ignore
- Rotary embeddings and QK-Norm, when enabled, are applied after _split_heads,
  and to the queries and keys only
- Attention weights are returned averaged over the heads, shape (N, L, S)
'''

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention.
    The torch counterpart of the mytorch.nn.MultiHeadAttention built in HW3P1.
    '''
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 rope: Optional[RotaryPositionalEncoding] = None, qk_norm: bool = False):
        '''
        Initialize the MultiHeadAttention.
        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate applied to the attention weights.
            rope (Optional[RotaryPositionalEncoding]): Rotary embeddings, or None to disable.
            qk_norm  (bool): Whether to normalize the queries and keys before scoring.
        '''
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY THESE ATTRIBUTES
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.rope      = rope
        self.q_norm    = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm    = RMSNorm(self.head_dim) if qk_norm else None

        # Initialize the dropout applied to the attention weights
        self.dropout  = nn.Dropout(dropout)

        # Initialize the projections, each embed_dim -> embed_dim
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the MultiHeadAttention.
        Args:
            query (torch.Tensor): The queries. shape: (N, L, embed_dim)
            key (torch.Tensor): The keys. shape: (N, S, embed_dim)
            value (torch.Tensor): The values. shape: (N, S, embed_dim)
            key_padding_mask (Optional[torch.Tensor]): Positions to ignore. shape: (N, S)
            attn_mask (Optional[torch.Tensor]): Positions to ignore. shape: (L, S)

        Returns:
            output (torch.Tensor): The output tensor. shape: (N, L, embed_dim)
            attn_weights (torch.Tensor): The attention weights, averaged over
                                         heads. shape: (N, L, S)
        '''
        N, L, _ = query.shape
        S       = key.shape[1]

        # Project the inputs
        q = self.q_proj(query)  # (N, L, E)
        k = self.k_proj(key)    # (N, S, E)
        v = self.v_proj(value)  # (N, S, E)

        # Reshape for multiple heads
        q = self._split_heads(q)  # (N, H, L, head_dim)
        k = self._split_heads(k)  # (N, H, S, head_dim)
        v = self._split_heads(v)  # (N, H, S, head_dim)

        # Apply QK-Norm and rotary embeddings to the queries and keys, if enabled
        # (never to the values)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Combine the padding and attention masks
        mask = self._merge_masks(key_padding_mask, attn_mask, N, L, S)  # broadcastable to (N, H, L, S)

        # Score, mask, normalize and apply dropout to get the attention weights
        # Scale the scores by the square root of head_dim before masking
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, H, L, S)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)  # (N, H, L, S)
        attn_weights = self.dropout(attn_weights)

        # Apply the attention weights to the values
        attn_output = torch.matmul(attn_weights, v)  # (N, H, L, head_dim)

        # Merge the heads
        attn_output = self._concat_heads(attn_output)  # (N, L, E)

        # Final projection, and average the attention weights over the heads
        output = self.out_proj(attn_output)  # (N, L, E)
        return output, attn_weights.mean(dim=1)  # (N, L, E), (N, L, S)

    def _merge_masks(self, key_padding_mask: Optional[torch.Tensor],
                     attn_mask: Optional[torch.Tensor], N: int, L: int, S: int) -> Optional[torch.Tensor]:
        '''
        Merge the two mask types into a single mask.
        Args:
            key_padding_mask (Optional[torch.Tensor]): Positions to ignore. shape: (N, S)
            attn_mask (Optional[torch.Tensor]): Positions to ignore. shape: (L, S)
            N (int): The batch size.
            L (int): The query sequence length.
            S (int): The key sequence length.
        Returns:
            Optional[torch.Tensor]: The combined mask, broadcastable to
                                    (N, num_heads, L, S), or None if both were None.
        '''
        if key_padding_mask is None and attn_mask is None:
            return None

        merged = None

        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S): the same padded keys are hidden for every head and query
            merged = key_padding_mask[:, None, None, :]

        if attn_mask is not None:
            # (L, S) -> (1, 1, L, S): the same pattern for every sequence and head
            expanded = attn_mask[None, None, :, :]
            merged = expanded if merged is None else torch.logical_or(merged, expanded)  # (N, 1, L, S)

        return merged

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Split the last dimension into (num_heads, head_dim).
        Args:
            x (torch.Tensor): Input tensor. shape: (N, L, embed_dim)
        Returns:
            torch.Tensor: Reshaped tensor. shape: (N, num_heads, L, head_dim)
        '''
        N, T, E = x.shape
        x = x.reshape(N, T, self.num_heads, self.head_dim)  # (N, T, H, head_dim)
        return x.transpose(1, 2)                             # (N, H, T, head_dim)

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Concatenate the heads back into the embedding dimension.
        Args:
            x (torch.Tensor): Input tensor. shape: (N, num_heads, L, head_dim)
        Returns:
            torch.Tensor: Reshaped tensor. shape: (N, L, embed_dim)
        '''
        N, H, T, d = x.shape
        x = x.transpose(1, 2)           # (N, T, H, head_dim)
        return x.reshape(N, T, H * d)   # (N, T, E), reshape copies since the transpose is non-contiguous
