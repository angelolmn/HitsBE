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
    np.testing.assert_array_equal(X_adj[start:start + len(X)], X)
    # And that the rest of the array is zeros
    assert np.all(X_adj[:start] == 0), "Values before the centered segment should be 0"
    assert np.all(X_adj[start + len(X):] == 0), "Values after the centered segment should be 0"

def test_crossCorr(hitsbe_instance):
    """
    Test the crossCorr() function with a series made by repeating a word.
    In this case, each segment should match the word exactly, yielding a correlation of 1.0.
    """
    word = np.linspace(0, 1, hitsbe_instance.cell_size)
    # Build a series by repeating the word for model.dim_seq segments
    segments = [word for _ in range(hitsbe_instance.dim_seq)]
    serie = np.concatenate(segments)
    
    # Using the method crossCorr from Hitsbe
    correlations = Hitsbe.crossCorr(serie, word)
    for corr in correlations:
        np.testing.assert_almost_equal(corr, 1.0, decimal=5,
            err_msg="Each correlation should be approximately 1.0")

def test_get_word(hitsbe_instance):
    """
    Test that get_word() returns segments with correlations above the threshold.
    We build a series where each segment equals the word exactly.
    """
    word = np.linspace(0, 1, hitsbe_instance.cell_size)
    segments = [word for _ in range(hitsbe_instance.dim_seq)]
    serie = np.concatenate(segments)
    
    # Set a threshold to capture the exact matches
    hitsbe_instance.threshold = 0.9
    result = hitsbe_instance.get_word(serie, word)
    
    # Obtain correlations using crossCorr() for reference
    correlations = Hitsbe.crossCorr(serie, word)
    
    # Check that returned segments have correlation above the threshold and valid indices
    for seg_idx, corr in result:
        assert abs(corr) > hitsbe_instance.threshold, "Correlation should exceed the threshold"
        assert 0 <= seg_idx < len(correlations), "Segment index out of expected range"

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
    Test that compute_word_embedding() returns the correct number of embeddings,
    and each embedding has the expected dimension
    """
    # Create a dummy sequence and a mask
    # For testing, we mark three segments as valid
    dummy_seq = [(0, 0.9), (1, 0.95), (0, 0.0), (1, 0.8)]
    seq_mask = np.array([1, 1, 0, 1])
    
    word_embeddings = hitsbe_instance.compute_word_embedding(seq_mask, dummy_seq)
    
    # Check that the number of embeddings equals the number of valid segments
    assert len(word_embeddings) == np.sum(seq_mask)
    
    # Each embedding should be a tensor with shape (dim_model,)
    for emb in word_embeddings:
        assert emb.shape == (hitsbe_instance.dim_model,)


def test_compute_positional_embedding(hitsbe_instance):
    """
    Test that compute_positional_embedding() returns the positional embeddings 
    for only the valid positions as indicated by the mask
    """
    # Create a dummy mask of length equal to the number of segments
    seq_mask = np.array([1, 0, 1, 1, 0])
    pos_embeddings = hitsbe_instance.compute_positional_embedding(seq_mask)
    
    # Check that the number of returned embeddings equals the number of valid positions
    assert len(pos_embeddings) == np.sum(seq_mask)
    
    # Check that each positional embedding has the expected dimension
    for emb in pos_embeddings:
        assert emb.shape == (hitsbe_instance.dim_model,)


def test_compute_haar_embedding(hitsbe_instance):
    """
    Test that compute_haar_embedding() returns an embedding for each valid segment
    For this test, we create dummy Haar coefficients arrays
    """
    # Create a dummy mask for the segments
    seq_mask = np.zeros(hitsbe_instance.dim_seq, dtype=int)
    # Mark every other segment as valid
    seq_mask[::2] = 1

    # For testing, choose a fixed number of Haar levels (e.g., 3)
    n_levels = 3
    # Adjust nhaar_level to match our test
    hitsbe_instance.nhaar_level = n_levels
    # Create dummy Haar coefficients
    # Each level is an array with length = dim_seq // n_levels
    haar_coeffs = []
    for level in range(n_levels):
        level_length = hitsbe_instance.dim_seq // n_levels
        # Fill with a constant value (e.g., level+1) for simplicity
        haar_coeffs.append(np.full(level_length, level + 1.0))
    
    # Adjust the Haar embedding matrix shape accordingly.
    hitsbe_instance.haar_emb_matrix = nn.Parameter(torch.randn(n_levels, hitsbe_instance.dim_model))
    
    haar_embedding = hitsbe_instance.compute_haar_embedding(seq_mask, haar_coeffs)
    
    # The expected shape of haar_embedding is (number_of_valid_segments, dim_model)
    n_valid = int(np.sum(seq_mask))
    assert haar_embedding.shape == (n_valid, hitsbe_instance.dim_model)


def test_get_embedding(hitsbe_instance):
    """
    Test that get_embedding() returns a concatenation of word, positional, and Haar embeddings
    We override get_sequence() to return a known dummy sequence
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
    
    embedding = hitsbe_instance.get_embedding(X)
    
    # It is expected that for each segment, the word, positional and Haar embeddings are summed,
    # so the final number of embedding items should equal dim_seq
    expected_items = hitsbe_instance.dim_seq
    assert isinstance(embedding, list)
    assert len(embedding) == expected_items

