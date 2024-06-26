class FeedForward(nn.Module):

    def __init__(self, embedding_dim, ff_dim):
        """
        Initialize the feed forward layer.

        Args:
            emb_dim (int) : The dimension of the embedding.
            ff_dim (int) : The dimension of the feed forward layer.
            dropout_rate (float) : The dropout rate. (default: 0.2)
        """
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_dim, embedding_dim)


    def forward(self, x):
        """
        Forward pass of the feed forward layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, embedding_dim, head_dim, context_size, ff_dim):
        """
        Initialize the decoder layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            head_dim (int): Dimensionality of each head.
            context_size (int): Size of the context window.
            ff_dim (int): Dimensionality of the feed-forward layer.
        """
        super().__init__()
        self.attention = MultiAttentionBlock(embedding_dim, head_dim, context_size)
        self.feed_forward = FeedForward(embedding_dim, ff_dim)
        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x):
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        x_norm = self.norm_1(x)
        attention = self.attention(x_norm)
        attention = attention + x

        attention_norm = self.norm_2(attention)
        ff = self.feed_forward(attention_norm)
        ff = ff + attention

        return ff