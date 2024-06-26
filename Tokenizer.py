class Tokenizer:

    @staticmethod
    def create_vocab(dataset):
        """
        Create a vocabulary from a dataset.

        Args:
            dataset (str): Text dataset to be used to create the character vocab.

        Returns:
            Dict[str, int]: Character vocabulary.
        """
        vocab = {
            token: index
            for index, token in enumerate(sorted(list(set(dataset))))
        }

        # Adding unknown token
        vocab["<unk>"] = len(vocab)

        return vocab

    def __init__(self, vocab):
        """
        Initialize the tokenizer.

        Args:
            vocab (Dict[str, int]): Vocabulary.
        """
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        self.vocab_decode = {v: k for k, v in self.vocab_encode.items()}

    def encode(self, text):
        """
        Encode a text in level character.

        Args:
            text (str): Input text to be encoded.

        Returns:
            List[int]: List with token indices.
        """
        return [self.vocab_encode.get(char, self.vocab_encode["<unk>"]) for char in text]

    def decode(self, indices):
        """
        Decode a list of token indices.

        Args:
            indices (List[int]): List of token indices.

        Returns:
            str: The decoded text.
        """
        return "".join([self.vocab_decode.get(idx, "<unk>") for idx in indices])