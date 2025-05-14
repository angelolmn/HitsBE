import numpy as np
from scipy.interpolate import make_interp_spline

import os

class Vocabulary:
    def __init__(self, file = "", domain_len = 8):
        
        # Initialize a list to store the words by monotone indexation
        self.words = [[] for _ in range(2**(domain_len-1))]

        self.domain_len = domain_len
        
        if file=="":
            self.primal_vocab()
        else:
            if os.path.exists(file):
                self.load_words(file)
            else:
                raise FileNotFoundError(f"File '{file}' not found.")


    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        size = 0
        for w in self.words:
            size += len(w)

        return size
    
    def __getitem__(self, key):
        return self.words[key]

    def _validate_word(self, word):
        """
        Validates that the word has exactly domain_len elements and that each element is a float in the range [0, 1].
        """
        if not isinstance(word, list):
            raise ValueError("The word must be a list of numbers.")
        if len(word) != self.domain_len:
            raise ValueError("The word must have exactly {dom} elements.")
        if not all(isinstance(value, float) for value in word):
            raise ValueError("All elements of the word must be floats.")
        if not all(0 <= value <= 1 for value in word):
            raise ValueError("All elements of the word must be in the range [0, 1].")
        
    def _compute_index(self, word):
        index = ""
        
        for i in range(len(word) - 1):
            if word[i] > word[i+1]:
                index += "0"
            else:
                index += "1"

        return int(index, 2)

    def load_words(self, file):
        with open(file, "r") as f:
            vocab_size_line = f.readline()
            vocab_size = int(vocab_size_line)

            for _ in range(vocab_size):
                word_line = f.readline()
                word = list(map(float, word_line.split()))
                self.add(word)

    def save_words(self,savefile):
        vocab_size = 0
        for w in self.words:
            vocab_size += len(w)

        with open(savefile, "w") as f:
            f.write(str(vocab_size) + "\n")
            for word_list in self.words:
                for word in word_list:
                    new_line = " ".join(map(str, word))
                    f.write(new_line + "\n")

    def add(self, word):
        """
        Adds a new word to the vocabulary after validating it.
        """
        self._validate_word(word)
        index = self._compute_index(word)
        self.words[index].append(word)

    def delete(self, word):
        """
        Deletes a word from the vocabulary
        """
        try:
            index = self._compute_index(word)
            self.words[index].remove(word)
        except ValueError:
            raise ValueError("The provided word is not in the vocabulary.")

    def primal_vocab(self):
        # x, x^2, x^3
        # 3x/2, 3x^2/2, 3x^3/2
        # x/2, x^2/2, x^3/2
        # x/4, x^2/4, x^3/4 
        domain = np.linspace(0,1,self.domain_len)
        div = np.linspace(1/4,1,4)

        for i in range(1,4):
            for j in div:
                self.add(((domain**i)*j).tolist())

        # x^(1/2), x^(1/3), x^(1/8)
        # x^(1/2)/2, x^(1/3)/2, x^(1/8)/2
        
        for i in [2,3,8]:
            self.add((domain**(1/i)).tolist())            
            self.add((domain**(1/i)/2).tolist())
        
        # y = 1/2
        self.add([1/2 for _ in range(self.domain_len)])        
        # -4*(x-1)x
        self.add((-4*domain*(domain-1)).tolist())

        #15x(x-3/4)^2
        self.add((15*domain*((domain-3/4)**2)).tolist())    

        # (sen({2,4}*PI*x)+1)/2
        # (cos({2,4}*PI*x)+1)/2

        for i in [2,4]:
            self.add(((np.sin(i*np.pi*domain) + 1)/2).tolist())    
            self.add(((np.cos(i*np.pi*domain) + 1)/2).tolist())    
            

    def unif_vocab(self, size):
        np.random.seed(42)

        for _ in range(size):
            self.add(np.random.rand(self.domain_len).tolist())

