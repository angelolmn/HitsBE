import numpy as np
import pytest
import torch
import torch.nn as nn

from hitsbe import Hitsbe


@pytest.fixture
def hitsbe_instance():
    # Create an instance of Hitsbe.
    instance = Hitsbe(primal=True)
    return instance


def test_adjust(hitsbe_instance):
    """
    Test that _adjust() centers a shorter array X in an array of length self.size.
    """
    X = np.arange(100)  # An array of length 100
    X_adj = hitsbe_instance._adjust(X)
    assert len(X_adj) == hitsbe_instance.size, "Adjusted array length should equal hitsbe.size"
    
    # Calculate expected start index
    start = (hitsbe_instance.size - len(X)) // 2
    # Check that the centered part is equal to X
    np.testing.assert_array_equal(X_adj[start:start + len(X)].cpu().numpy(), X)
    # And that the rest of the array is zeros
    assert np.all(X_adj[:start].cpu().numpy() == 0), "Values before the centered segment should be 0"
    assert np.all(X_adj[start + len(X):].cpu().numpy() == 0), "Values after the centered segment should be 0"


def test_cross_corr(hitsbe_instance):
    """
    Test the cross_corr() function with a series made by repeating a word.
    In this case, each segment should match the word exactly, yielding a correlation of 1.0.
    """
    word = np.linspace(0, 1, hitsbe_instance.cell_size)
    # Build a series by repeating the word for dim_seq segments
    segments = [word for _ in range(hitsbe_instance.dim_seq)]
    serie = np.concatenate(segments)
    
    correlations = Hitsbe.cross_corr(serie, word)
    for corr in correlations:
        np.testing.assert_almost_equal(corr, 1.0, decimal=5,
            err_msg="Each correlation should be approximately 1.0")


def test_get_sequence(hitsbe_instance):
    """
    Test that get_sequence() returns a sequence with length equal to hitsbe.dim_seq,
    and that each element is a tuple (segment_index, correlation).
    """
    word = np.linspace(0, 1, hitsbe_instance.cell_size)
    segments = [word for _ in range(hitsbe_instance.dim_seq)]
    serie = np.concatenate(segments)
    
    # Lower threshold to ensure words are matched
    hitsbe_instance.threshold = 0.5
    seq = hitsbe_instance.get_sequence(serie)
    
    assert len(seq) == hitsbe_instance.dim_seq, "Sequence length should equal hitsbe.dim_seq"
    for segment_index, corr in seq:
        assert isinstance(segment_index, int), "Segment index should be an integer"
        assert isinstance(corr, float), "Correlation value should be a float"


def test_compute_word_embedding(hitsbe_instance):
    """
    Test that compute_word_embedding() returns an embedding for each segment,
    and each embedding has the expected dimension.
    For segments marked as invalid (mask == 0), the embedding should be a zero vector.
    """
    # Create a dummy sequence and a mask
    dummy_seq = [(0, 0.9), (1, 0.95), (0, 0.0), (1, 0.8)]
    seq_mask = np.array([1, 1, 0, 1])
    
    word_embeddings = hitsbe_instance.compute_word_embedding(seq_mask, dummy_seq)
    
    # Check that the number of embeddings equals the sequence length
    assert len(word_embeddings) == len(seq_mask), "Number of embeddings should equal the sequence length"
    
    # Each embedding should be a tensor with shape (dim_model,)
    for i, emb in enumerate(word_embeddings):
        assert emb.shape == (hitsbe_instance.dim_model,), "Each embedding should have shape (dim_model,)"
        # For invalid segments (mask == 0), the embedding should be all zeros.
        if seq_mask[i] == 0:
            assert torch.all(emb == 0), "Invalid segment should return a zero embedding"



def test_compute_haar_embedding(hitsbe_instance):
    """
    Test that compute_haar_embedding() returns an embedding for each segment,
    where valid segments have a computed embedding and invalid segments are zero vectors.
    """
    # Create a dummy mask for the segments of length equal to dim_seq
    seq_mask = np.zeros(hitsbe_instance.dim_seq, dtype=int)
    # Mark every other segment as valid
    seq_mask[::2] = 1

    # For testing, choose a fixed number of Haar levels (e.g., 3)
    n_levels = 3
    # Adjust nhaar_level to match our test
    hitsbe_instance.nhaar_level = n_levels
    # Create dummy Haar coefficients for each level
    # Each level is an array with length = dim_seq // n_levels
    haar_coeffs = []
    for level in range(n_levels):
        level_length = hitsbe_instance.dim_seq // n_levels
        # Fill with a constant value (e.g., level+1) for simplicity
        haar_coeffs.append(np.full(level_length, level + 1.0))
    
    # Adjust the Haar embedding matrix shape accordingly.
    hitsbe_instance.haar_emb_matrix = nn.Parameter(torch.randn(n_levels, hitsbe_instance.dim_model))
    
    haar_embedding = hitsbe_instance.compute_haar_embedding(seq_mask, haar_coeffs)
    
    # The expected shape of haar_embedding is (dim_seq, dim_model)
    assert haar_embedding.shape == (hitsbe_instance.dim_seq, hitsbe_instance.dim_model), "Haar embedding shape mismatch"
    
    # For invalid segments (mask == 0), the embedding should be a zero vector.
    for i, m in enumerate(seq_mask):
        if m == 0:
            assert torch.all(haar_embedding[i] == 0), "Invalid segment should return a zero Haar embedding"


def test_get_embedding(hitsbe_instance):
    """
    Test that get_embedding() returns a concatenation of word, positional, and Haar embeddings.
    We override get_sequence() to return a known dummy sequence.
    """
    # Create a dummy time series X of the required length
    X = np.linspace(0, 1, hitsbe_instance.size)
    
    # Create a dummy sequence: all segments are valid
    dummy_seq = [(i % len(hitsbe_instance.vocabulary), 1.0) for i in range(hitsbe_instance.dim_seq)]
    # Override get_sequence() to return our dummy sequence
    hitsbe_instance.get_sequence = lambda X: dummy_seq
    
    # To ensure consistency, also override the Haar decomposition
    # For simplicity, let’s simulate three levels
    n_levels = 3
    hitsbe_instance.nhaar_level = n_levels
    # Adjust the Haar embedding matrix accordingly.
    hitsbe_instance.haar_emb_matrix = nn.Parameter(torch.randn(n_levels, hitsbe_instance.dim_model))
    
    # Wrap X in a list so that get_embedding iterates over a batch of one time series
    embedding = hitsbe_instance.get_embedding([X])
    
    # It is expected that for each segment, the word, positional and Haar embeddings are summed,
    # so the final number of embedding items for each time series should equal dim_seq.
    expected_items = hitsbe_instance.dim_seq
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 1, "There should be one embedding per input time series"
    # embedding[0] is a list of segment embeddings, so we check its length.
    assert len(embedding[0]) == expected_items, "Each embedding list should have dim_seq segments"
