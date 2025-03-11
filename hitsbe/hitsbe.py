import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import pywt 
import math

from hitsbe import Vocabulary


class Hitsbe():
    def __init__(self, primal = True):
        self.vocabulary = Vocabulary(primal)
        self.threshold = 0.95
        self.size = 1024
        self.cell_size = 8
        self.dim_seq = self.size // self.cell_size

        self.att_mask = torch.ones(self.dim_seq, dtype=torch.int32)

        # 768 for BERT
        self.dim_model = 768

        # Number of levels we will have in the Haar transform
        # associated with the sequence 
        self.nhaar_level = int(np.log2(self.dim_seq)) + 1

        self.word_emb_matrix = nn.Embedding(len(self.vocabulary), self.dim_model)
        #self.word_emb_matrix = nn.Parameter(torch.randn(len(self.vocabulary.words), 768))
        self.haar_emb_matrix = nn.Parameter(torch.randn(int(self.nhaar_level), self.dim_model))

        # Positional encoding.
        # The sinusoidal version is the simplest to implement initially.
        # We currently plan to study Haar indexing, so learned positional embeddings
        # are reserved for future work.

        # https://discuss.pytorch.org/t/positional-encoding/175953
        self.pos_emb_matrix = torch.zeros(self.dim_seq, self.dim_model)
        position = torch.arange(0, self.dim_seq, dtype=torch.float).unsqueeze(1)
        # exp(a*ln(b)) = b^a 
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model))
        self.pos_emb_matrix[:, 0::2] = torch.sin(position * div_term)
        self.pos_emb_matrix[:, 1::2] = torch.cos(position * div_term)

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

        # Config att_mask
        start_att_mask = start//self.cell_size
        end_att_mask = (start + n)//self.cell_size

        self.att_mask[:start_att_mask] = 0
        self.att_mask[end_att_mask + 1:] = 0 

        return X_adj
        
    @staticmethod
    def cross_corr(serie, word):
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
            if np.std(word) == 0 or np.std(subseq) == 0:
                r = 0.0 
            else:
                r, _ = stats.pearsonr(word, subseq) # Calculate the Pearson correlation
            correlations.append(float(r)) # Store the correlation
            
        return correlations


    def get_sequence(self, X):
        """
        Returns the sequence with the corresponding words and their correlations.
        """
        seq = [(-np.inf, -np.inf) for _ in range(self.dim_seq)]

        if len(X) != self.size:
            X = self._adjust(X)
        
        # Fill the sequence with the corresponding words
        for i, w in enumerate(self.vocabulary):
            corr = self.cross_corr(X, w)
            for j, c in enumerate(corr):
                if seq[j][1] < abs(c) and abs(c) > abs(self.threshold):
                    seq[j] = (i, c)


        return seq

    def compute_word_embedding(self, seq_mask, seq):
        """
        Computes the word embeddings for each segment indicated as valid by the sequence mask
        """
        word_emb = [] 
        
        # Iterate over each element in seq_mask along with its index
        for i, m in enumerate(seq_mask):
            # If the mask value is truthy (e.g., 1 or True), then the segment is valid
            if m:
                # Retrieve the word embedding for the segment:
                # seq[i][0] is used as an index into the word embedding module
                idx = seq[i][0]
                # Call the embedding module with a tensor index and remove the extra batch dimension
                emb = self.word_emb_matrix(torch.tensor([idx])).squeeze(0)
                word_emb.append(emb)

            else:
                word_emb.append(torch.zeros(self.dim_model,
                                            device=self.word_emb_matrix.weight.device, 
                                            dtype=self.word_emb_matrix.weight.dtype)
                                )   
        
        # Return the list of computed word embeddings
        return word_emb 


    def compute_haar_embedding(self, seq_mask, haar_coeffs):
        """
        Computes the Haar embedding for each segment indicated as valid by the sequence mask
        """
        haar_coeff_to_embed = []
        
        for i, m in enumerate(seq_mask):
            if m:  
                hc = [] 
                # For each level in the Haar coefficients,
                # append the corresponding value to haar_coeff_to_embed.
                #
                # If i is the segment index, d is the dimension of the sequence
                # (d is equal to the number of coefficients in our last
                # level of the Haar transform calculated) and n is the number
                # of coefficients in the current Haar level,
                # then the coefficient associated with the word is
                #                  i // (d / n)
                # Note: d mod n = 0
                for hlevel in range(self.nhaar_level):
                    index = int(i // (self.dim_seq / len(haar_coeffs[hlevel])))
                    # Append the coefficient from the current Haar level
                    hc.append(haar_coeffs[hlevel][index])

                # Append the list of coefficients for this valid segment
                haar_coeff_to_embed.append(hc)
            
            else:
                haar_coeff_to_embed.append(np.zeros(self.nhaar_level, dtype=np.float32))   
        
        # Convert the list of embeddings to a numpy array
        haar_coeff_to_embed = np.stack(haar_coeff_to_embed, axis=0)
        haar_emb_matrix_np = self.haar_emb_matrix.detach().cpu().numpy()
        # Compute the dot product between the matrix of Haar coefficients and the Haar embedding matrix
        haar_embed = np.dot(haar_coeff_to_embed, haar_emb_matrix_np)
        
        return haar_embed


    def get_embedding(self, X):
        
        final_embeddings = []

        for x in X:
            if len(x) != self.size:
                x = self._adjust(x)

            # Get the sequence (segments and correlations)
            seq = self.get_sequence(x)
            seq_mask = np.array([1 if w != (0, 0.0) else 0 for w in seq])

            word_embed = self.compute_word_embedding(seq_mask, seq)
            

            # Compute the Haar wavelet decomposition of x and select levels 1 to nhaar_level
            haar_coeffs = pywt.wavedec(x, 'haar')[1:int(self.nhaar_level)+1]

            # compute_haar_embedding returns a numpy array of shape (n_valid, dim_model)
            haar_embed_np = self.compute_haar_embedding(seq_mask, haar_coeffs)
            # Convert each row of the numpy array into a tensor (to be compatible with word and positional embeddings)
            haar_embed = [torch.tensor(row, dtype=word_embed[0].dtype) for row in haar_embed_np]

            # For each segment, we add the word, positional and Haar embeddings.
            final_embeddings.append([w + p + h for w, p, h in zip(word_embed, self.pos_emb_matrix, haar_embed)])

        return final_embeddings

