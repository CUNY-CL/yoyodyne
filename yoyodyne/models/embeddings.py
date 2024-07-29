"""Embedding initialization functions."""

from torch import nn


def xavier_embedding(
    num_embeddings: int, embedding_size: int, pad_idx: int
) -> nn.Embedding:
    """Initializes the embeddings layer using Xavier initialization.

    The pad embeddings are also zeroed out.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_size (int): dimension of embeddings.
        pad_idx (int): index of pad symbol.

    Returns:
        nn.Embedding: embedding layer.
    """
    embedding_layer = nn.Embedding(num_embeddings, embedding_size)
    # Xavier initialization.
    nn.init.normal_(embedding_layer.weight, mean=0, std=embedding_size**-0.5)
    # Zeroes out pad embeddings.
    if pad_idx is not None:
        nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
    return embedding_layer


def normal_embedding(
    num_embeddings: int, embedding_size: int, pad_idx: int
) -> nn.Embedding:
    """Initializes the embeddings layer from a normal distribution.

    The pad embeddings are also zeroed out.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_size (int): dimension of embeddings.
        pad_idx (int): index of pad symbol.

    Returns:
        nn.Embedding: embedding layer.
    """
    embedding_layer = nn.Embedding(num_embeddings, embedding_size)
    # Zeroes out pad embeddings.
    if pad_idx is not None:
        nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
    return embedding_layer
