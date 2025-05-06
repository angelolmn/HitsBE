import numpy as np
from scipy.interpolate import make_interp_spline

import os

class Vocabulary:
    def __init__(self, file = "", size = 100):
        
        # Initialize a list to store the words
        self.words = []
        
        if file=="":
            self.primal_vocab()
            self.spline_vocab(size)
        else:
            if os.path.exists(file):
                self._load_words(file)
            else:
                raise FileNotFoundError(f"File '{file}' not found.")


    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

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

    def _load_words(self, file):
        with open(file, "r") as f:
            vocab_size_line = f.readline()
            vocab_size = int(vocab_size_line)

            for _ in range(vocab_size):
                word_line = f.readline()
                word = list(map(float, word_line.split()))
                self.add(word)

    def save_words(self,savefile):
        vocab_size = len(self.words)
        with open(savefile, "w") as f:
            f.write(str(vocab_size) + "\n")
            for word in self.words:
                new_line = " ".join(map(str, word))
                f.write(new_line + "\n")

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

    def primal_vocab(self):
        # x, x^2, x^3
        # 3x/2, 3x^2/2, 3x^3/2
        # x/2, x^2/2, x^3/2
        # x/4, x^2/4, x^3/4 
        domain_len = 8
        domain = np.linspace(0,1,domain_len)
        div = np.linspace(1/4,1,4)

        for i in range(1,4):
            for j in div:
                self.words.append((domain**i)*j)

        # x^(1/2), x^(1/3), x^(1/8)
        # x^(1/2)/2, x^(1/3)/2, x^(1/8)/2
        
        for i in [2,3,8]:
            self.words.append(domain**(1/i))
            self.words.append(self.words[-1]/2)
        
        # y = 1/2
        self.words.append(np.full(domain_len, 1/2))
        
        # -4*(x-1)x
        self.words.append(-4*domain*(domain-1))

        #15x(x-3/4)^2
        self.words.append(15*domain*((domain-3/4)**2))

        # (sen({2,4}*PI*x)+1)/2
        # (cos({2,4}*PI*x)+1)/2

        for i in [2,4]:
            self.words.append(
                (np.sin(i*np.pi*domain) + 1)/2
                )
            self.words.append(
                (np.cos(i*np.pi*domain) + 1)/2
                )
            
        np.random.seed(42)

        # Add noise N(0; 0,05^2)
        noise_words = []
        for w in self.words: 
            noise_words.append(np.clip(w + np.random.normal(0, 0.05, size=domain_len),0,1))  

        for w in noise_words:
            self.words.append(w)

    def spline_vocab(self, size):
        domain_len = 8

        np.random.seed(42)
        xs = np.linspace(0, 1, 5)
        domain = np.linspace(0,1,domain_len)

        for _ in range(size):
            ys = np.random.rand(5)
            spline = make_interp_spline(xs, ys, k=3)

            word = spline(domain)

            wmin = np.min(word)
            wmax = np.max(word)

            norm_word = (word - wmin)/(wmax -wmin)

            self.words.append(norm_word)

