import torch
import torch.nn as nn

# Token Embedding layer mapping discrete token IDs to dense vectors
# Example:
#   Input token_ids: shape (B, T) e.g., (8, 128)
#   Output embedding: shape (B, T, C) e.g., (8, 128, 256)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # Trainable lookup table of shape (vocab_size, embed_size)
        # e.g., mapping 65 unique tokens to 256-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, token_ids):
        # Maps (B, T) token IDs to (B, T, C) vector embeddings
        return self.embedding(token_ids)
