import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import pywt 
import math

from hitsbe import Vocabulary


class Hitsbe:
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.threshold = 0.9
        self.size = 1024
        self.cell_size = 8
        self.dim_seq = self.size // self.cell_size

        # 768 for BERT
        self.dim_model = 768

        # Number of levels we will have in the Haar transform
        # associated with the sequence 
        self.nhaar_level = np.log2(self.dim_seq) + 1 

        self.word_emb_matrix = nn.Embedding(len(self.vocabulary.words), self.dim_model)
        #self.word_emb_matrix = nn.Parameter(torch.randn(len(self.vocabulary.words), 768))
        self.haar_emb_matrix = nn.Parameter(torch.randn(self.nhaar_level, self.dim_model))

        # Positional encoding.
        # The sinusoidal version is the simplest to implement initially.
        # We currently plan to study Haar indexing, so learned positional embeddings
        # are reserved for future work.

        # https://discuss.pytorch.org/t/positional-encoding/175953
        self.pe = torch.zeros(self.dim_seq, self.dim_model)
        position = torch.arange(0, self.dim_seq, dtype=torch.float).unsqueeze(1)
        # exp(a*ln(b)) = b^a 
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # self.size mod self.cell_size == 0
        # log_2(self.size) = int

    def _adjust(self, X):
        "Centers the array X"
        X_adj = np.zeros(self.size)
        n = len(X)
        
        # Calculate the starting index so that X is centered
        start = (self.size - n) // 2

        # Create a zero array with the same data type as X
        X_adj = np.zeros(self.size, dtype=X.dtype)
        
        # Place X into the center of X_adj
        X_adj[start:start + n] = X
        return X_adj
        
    
    def crossCorr(serie, word):
        """
        Computes the correlation of the word with each segment of the time series of the same size.
        Returns a list of floating-point correlations.
        """
        serie = np.array(serie, dtype=np.float32)
        word = np.array(word, dtype=np.float32)

        N = len(serie)  # Series length (ideally 2^10)
        M = len(word)   # Word length (ideally 2^3)          N / M = 128 = 2^7
        
        correlations = []  # Will store the Pearson coefficients

        for i in range(0, N - M + 1, M):
            subseq = serie[i:i + M]  # Extract the subsequence of size M
            r, _ = stats.pearsonr(word, subseq)  # Calculate Pearson correlation
            correlations.append(float(r))  # Store the correlation
            
        return correlations


    def get_word(self, serie, word):
        """
        Returns the segment where the word is found and its correlation.
        First, it computes the correlation of the word in each segment,
        then returns those segments whose absolute correlation exceeds a threshold.
        """
        corr = self.crossCorr(serie, word)
        # i is the segment of the series where the word is found
        # c is the correlation of the word in such segment
        return [(i, c) for i, c in enumerate(corr) if abs(c) > self.threshold]  


    def get_sequence(self, X):
        """
        Returns the sequence with the corresponding words and their correlations.
        """
        seq = [(0, 0.0) for _ in range(self.dim_seq)]

        if len(X) != self.size:
            X = self._adjust(X)
        
        # Fill the sequence with the corresponding words
        for i, w in enumerate(self.vocabulary):
            corr = self.get_word(X, w)  # Get segment and correlation for word w 
            for c in corr:
                # In a segment (i) only the word with the highest correlation should remain.
                if seq[i][1] < c[1]:
                    seq[i] = c

        return seq

        

    def get_embedding(self, X):
        # Get the sequence (segments and correlations)
        seq = self.get_sequence(X)

        embedding = []

        # Compute the Haar wavelet decomposition of X and select levels 1 to nhaar_level
        haar_coeffs = pywt.wavedec(X, 'haar')[1:self.nhaar_level+1]

        for s in seq:
            if s[1] > self.threshold:
                word_embed = self.word_emb_matrix[s[0]]

                haar_coeff_to_embed = []

                # For each level in the Haar coefficients,
                # append the corresponding value to haar_coeff_to_embed.
                #
                # If i is the word index, d is the dimension of the sequence
                # (d is equal to the number of coefficients in our last
                # level of the Haar transform calculated) and n is the number
                # of coefficients in the current Haar level,
                # then the coefficient associated with the word is
                #                  i // (d / n)
                # Note: d mod n = 0
                for hlevel in self.nhaar_level:
                    haar_coeff_to_embed.append(
                        haar_coeffs[hlevel][s[0] // (self.dim_seq / len(haar_coeffs))]
                    )

                haar_embed = np.dot(self.haar_emb_matrix, haar_coeff_to_embed)

                # Classical positional encoding
                pos_embed = self.pe[s[0]]

                embedding.append(word_embed + haar_embed + pos_embed)

        return embedding
