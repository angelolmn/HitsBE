import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import pywt 
import math

from hitsbe import Vocabulary


class Hitsbe(nn.Module):
    def __init__(self, primal=True):
        super(Hitsbe, self).__init__()
        self.vocabulary = Vocabulary(primal)
        self.threshold = 0.95
        self.size = 1024
        self.cell_size = 8
        self.dim_seq = self.size // self.cell_size

        self.register_buffer("att_mask", torch.ones(self.dim_seq, dtype=torch.int32))

        # 768 for BERT
        self.dim_model = 768

        # Number of levels for the Haar transform
        self.nhaar_level = int(np.log2(self.dim_seq)) + 1

        self.word_emb_matrix = nn.Embedding(len(self.vocabulary), self.dim_model)
        self.haar_emb_matrix = nn.Parameter(torch.randn(int(self.nhaar_level), self.dim_model))

        # Positional encoding.
        # The sinusoidal version is the simplest to implement initially.
        # We currently plan to study Haar indexing, so learned positional embeddings
        # are reserved for future work.

        # https://discuss.pytorch.org/t/positional-encoding/175953
        position = torch.arange(0, self.dim_seq, dtype=torch.float).unsqueeze(1)
        # exp(a*ln(b)) = b^a 
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model))
        pos_emb = torch.zeros(self.dim_seq, self.dim_model)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb_matrix", pos_emb)


    def _adjust(self, X):
        "Centers the array X"
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float)

        n = len(X)
        
        # Calculate the starting index so that X is centered
        start = (self.size - n) // 2

        # Create a zero array with the same data type as X
        X_adj = torch.zeros(self.size, dtype=X.dtype, device=X.device)
        
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
        if not torch.is_tensor(serie):
            serie = torch.tensor(serie, dtype=torch.float)
        else:
            serie = serie.float()
            
        if not torch.is_tensor(word):
            # Move the word tensor to the same device as the serie
            word = torch.tensor(word, dtype=torch.float, device=serie.device)
        else:
            word = word.float().to(serie.device)

        N = len(serie)  # Series length (ideally 2^10)
        M = len(word)   # Word length (ideally 2^3)          N / M = 128 = 2^7

        correlations = []  # Will store the Pearson coefficients

        std_word = torch.std(word, unbiased=False)
        mean_word = torch.mean(word)

        for i in range(0, N - M + 1, M):
            subseq = serie[i:i + M]  # Extract the subsequence of size M
            std_subseq = torch.std(subseq, unbiased=False)

            if std_word == 0 or std_subseq == 0:
                r = 0.0 
            else:
                mean_subseq = torch.mean(subseq)
                cov = torch.mean((word - mean_word) * (subseq - mean_subseq))
                r = (cov / (std_word * std_subseq)).item()

            correlations.append(float(r))
            
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
                # Create the index tensor on the same device as seq_mask
                idx_tensor = torch.tensor([idx], device=seq_mask.device)
                # Call the embedding module with a tensor index and remove the extra batch dimension
                emb = self.word_emb_matrix(idx_tensor).squeeze(0)

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
                haar_coeff_to_embed.append(torch.tensor(hc, dtype=torch.float, device=self.haar_emb_matrix.device))
        
            else:
                haar_coeff_to_embed.append(torch.zeros(self.nhaar_level, dtype=torch.float, device=self.haar_emb_matrix.device))

        # Convert the list of embeddings to a tensor
        haar_coeff_to_embed_tensor = torch.stack(haar_coeff_to_embed, dim=0)
        # Compute the dot product between the matrix of Haar coefficients and the Haar embedding matrix
        haar_embed = torch.matmul(haar_coeff_to_embed_tensor, self.haar_emb_matrix)        
        
        return haar_embed


    def get_embedding(self, X):
    
        final_embeddings = []

        for x in X:
            # Ensure x is a torch tensor so that we can use .device
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float)

            # Move x to the same device as the model parameters (e.g., GPU)
            x = x.to(self.word_emb_matrix.weight.device)

                
            if len(x) != self.size:
                x = self._adjust(x)

            # Get the sequence (segments and correlations)
            seq = self.get_sequence(x)
            seq_mask = torch.tensor([1 if np.isfinite(w[0]) else 0 for w in seq], dtype=torch.long, device=x.device)
    
            word_embed = self.compute_word_embedding(seq_mask, seq)
            
            # Compute the Haar wavelet decomposition of x and select levels 1 to nhaar_level
            haar_coeffs_tuple = pywt.wavedec(x.cpu().numpy(), 'haar')[1:int(self.nhaar_level)+1]
            haar_coeffs = [torch.tensor(arr, dtype=torch.float, device=self.haar_emb_matrix.device) for arr in haar_coeffs_tuple]
            # compute_haar_embedding returns a tensor of shape (dim_seq, dim_model)
            haar_embed_tensor = self.compute_haar_embedding(seq_mask, haar_coeffs)
            # Convert each row of the tensor into a tensor (to be compatible with word and positional embeddings)
            haar_embed = [row.clone().detach().to(x.device).type(word_embed[0].dtype) for row in haar_embed_tensor]

            # Move pos_emb_matrix to the same device as x
            pos_emb = self.pos_emb_matrix.to(x.device)

            # For each segment, sum the word, positional and Haar embeddings.
            instance_embeddings = [w + p + h for w, p, h in zip(word_embed, pos_emb, haar_embed)]
            # Stack embeddings for the current instance: shape (dim_seq, dim_model)
            final_embeddings.append(torch.stack(instance_embeddings, dim=0))
    
        # Stack all instance embeddings: shape (batch_size, dim_seq, dim_model)
        return torch.stack(final_embeddings, dim=0)



