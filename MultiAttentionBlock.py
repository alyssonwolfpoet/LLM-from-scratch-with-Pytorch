class MultiAttentionBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size):
        """
        Initialize the MultiAttentionBlock layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            num_heads (int): Number of attention heads.
            context_size (int): Size of the context window.
        """
        super().__init__()

        # Checking number of heads
        head_dim = embedding_dim // num_heads
        assert head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.attention = nn.ModuleList(modules=[AttentionBlock(embedding_dim, head_dim, context_size) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
    
    def forward(self, x):
        """
        Forward pass of the MultiAttentionBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
        
        Returns:
            torch.Tensor: New embedding representation of shape (batch_size, seq_len, embedding_dim).
        """
        out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
        x = self.linear(x)
        return x