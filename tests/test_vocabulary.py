import pytest
from hitsbe import Vocabulary

@pytest.fixture
def vocab():
    return Vocabulary()

@pytest.fixture
def valid_word():
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

@pytest.fixture
def another_valid_word():
    return [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

@pytest.fixture
def int_word():
    return [1, 2, 3, 4, 5, 6, 7, 8]

@pytest.fixture
def mixed_word():
    return [0.1, 0.2, 0.3, 0.4, 1, 0.6, 0.7, 0.8]

def test_add_valid_word(vocab, valid_word):
    vocab.add(valid_word)
    assert len(vocab.words) == 1
    assert vocab.words[0] == valid_word

def test_add_invalid_word_length(vocab):
    with pytest.raises(ValueError):
        vocab.add([0.1, 0.2])

def test_add_invalid_word_range(vocab):
    invalid_word = [0.1, 0.2, 0.3, 0.4, 1.1, 0.6, 0.7, 0.8]
    with pytest.raises(ValueError):
        vocab.add(invalid_word)

def test_add_invalid_word_type(vocab):
    with pytest.raises(ValueError):
        vocab.add("not a list")

def test_add_int_word_rejected(vocab, int_word):
    with pytest.raises(ValueError):
        vocab.add(int_word)

def test_add_mixed_word_rejected(vocab, mixed_word):
    with pytest.raises(ValueError):
        vocab.add(mixed_word)

def test_modify_valid(vocab, valid_word, another_valid_word):
    vocab.add(valid_word)
    vocab.modify(0, another_valid_word)
    assert vocab.words[0] == another_valid_word

def test_modify_invalid_index(vocab, valid_word, another_valid_word):
    vocab.add(valid_word)
    with pytest.raises(IndexError):
        vocab.modify(1, another_valid_word)

def test_modify_invalid_word(vocab, valid_word):
    vocab.add(valid_word)
    invalid_word = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        vocab.modify(0, invalid_word)

def test_delete_by_index(vocab, valid_word, another_valid_word):
    vocab.add(valid_word)
    vocab.add(another_valid_word)
    vocab.delete(index=0)
    assert len(vocab.words) == 1
    assert vocab.words[0] == another_valid_word

def test_delete_by_word(vocab, valid_word, another_valid_word):
    vocab.add(valid_word)
    vocab.add(another_valid_word)
    vocab.delete(word=valid_word)
    assert len(vocab.words) == 1
    assert vocab.words[0] == another_valid_word

def test_delete_invalid_index(vocab, valid_word):
    vocab.add(valid_word)
    with pytest.raises(IndexError):
        vocab.delete(index=5)

def test_delete_word_not_found(vocab, valid_word, another_valid_word):
    vocab.add(valid_word)
    with pytest.raises(ValueError):
        vocab.delete(word=another_valid_word)

def test_delete_no_parameter(vocab, valid_word):
    vocab.add(valid_word)
    with pytest.raises(ValueError):
        vocab.delete()
