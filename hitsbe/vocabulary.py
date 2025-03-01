class Vocabulary:
    def __init__(self):
        # Initialize a list to store the words.
        self.words = []

    def _validate_word(self, word):
        """
        Validates that the word has exactly 8 elements and that each element is a float in the range [0, 1].
        """
        if not isinstance(word, list):
            raise ValueError("The word must be a list of numbers.")
        if len(word) != 8:
            raise ValueError("The word must have exactly 8 elements.")
        if not all(isinstance(value, float) for value in word):
            raise ValueError("All elements of the word must be floats.")
        if not all(0 <= value <= 1 for value in word):
            raise ValueError("All elements of the word must be in the range [0, 1].")


    def add(self, word):
        """
        Adds a new word to the vocabulary after validating it.
        """
        self._validate_word(word)
        self.words.append(word)

    def modify(self, index, new_word):
        """
        Modifies the word at the specified index with 'new_word' after validating it.
        """
        if index < 0 or index >= len(self.words):
            raise IndexError("The index is out of range.")

        self._validate_word(new_word)
        self.words[index] = new_word

    def delete(self, index=None, word=None):
        """
        Deletes a word from the vocabulary. It can be removed by specifying either the index or the word.
        """
        if index is not None:
            if index < 0 or index >= len(self.words):
                raise IndexError("The index is out of range.")
            del self.words[index]
        elif word is not None:
            try:
                self.words.remove(word)
            except ValueError:
                raise ValueError("The provided word is not in the vocabulary.")
        else:
            raise ValueError("You must provide either an index or a word to delete.")


