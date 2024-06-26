class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the Embedding layer with Positional Encoding.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pe = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the Embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        word_emb = self.embedding(x)
        word_pe = self.pe(x)
        return word_emb + word_pe

class AttentionBlock(nn.Module):

    def __init__(self, embedding_dim, context_size):
        """
        Initialize the AttentionBlock layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            context_size (int): Size of the context window.
        """
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        ones = torch.ones(size=[context_size, context_size], dtype=torch.float)
        self.register_buffer(name="mask", tensor=torch.tril(input=ones)) # Triangular matrix
    
    def forward(self, x):
        """
        Forward pass of the AttentionBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: New embedding representation of shape (batch_size, seq_len, embedding_dim).
        """
        B, T, C = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        qk = query @ key.transpose(-2, -1) * C**-0.5
        attention = qk.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        attention = F.softmax(input=attention, dim=-1)

        out = attention @ value
        return out