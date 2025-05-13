import torch
import torch.nn as nn
import pywt 
import math

from .vocabulary2 import Vocabulary


class Hitsbe(nn.Module):
    def __init__(self, filename=""):
        super(Hitsbe, self).__init__()

        self.vocabulary = Vocabulary(file=filename)
        
        vocab_dist = torch.tensor([len(w) for  w in self.vocabulary])
        vocab_dist = torch.cumsum(vocab_dist, dim=0)
        vocab_dist = torch.roll(vocab_dist, shifts = 1)
        vocab_dist[0] = 0
        # It contains the cumulative sums of all the previous monotonicity classes
        self.register_buffer("vocabulary_dist", vocab_dist) # No trainable

        
        self.ts_len = 1024
        self.segment_len = 8

        self.dim_seq = self.ts_len // self.segment_len

        # 768 for BERT
        self.dim_model = 768
        
        # Number of levels for the Haar transform
        self.nhaar_level = int(torch.log2(torch.tensor(self.dim_seq, dtype=torch.float32)).item())
        
        self.word_emb_matrix = nn.Embedding(len(self.vocabulary), self.dim_model)
        self.haar_emb_matrix = nn.Parameter(torch.randn(self.nhaar_level + 1, self.dim_model))

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
        self.register_buffer("pos_emb_matrix", pos_emb) # No trainable


    def _adjust(self, X):
        "Centers each row of the batch X."
        assert isinstance(X, torch.Tensor), "Expected X to be a torch.Tensor"

        device = next(self.parameters()).device
        batch_size = X.size(0)

        X_adj = torch.zeros(batch_size, self.ts_len, device=device)
        att_mask = torch.zeros(batch_size, self.dim_seq, dtype=torch.long, device=device)        


        for i,x in enumerate(X):
            N = x.size(0)

            if N < self.ts_len:
                # Calculate the starting index so that X is centered
                start = (self.ts_len - N) // 2
                end = start + N

                # Place X into the center of X_adj
                X_adj[i, start:end] = x

                # Config att_mask
                start_att_mask = start//self.segment_len
                end_att_mask = (start + N)//self.segment_len # include the last segment

                att_mask[i, start_att_mask:end_att_mask] = 1
            elif N > self.ts_len:
                X_adj[i] = x[:self.ts_len]
                att_mask[i] = 1
            else:
                X_adj[i] = x
                att_mask[i] = 1
            
        return X_adj, att_mask
        
    def get_best_approach(self, segment, return_index = False):
        device = next(self.parameters()).device
        segment_bit_index = self.vocabulary._compute_index(segment)

        smin = torch.min(segment)
        smax = torch.max(segment)
        # (segment_len)
        segment_norm = (segment - smin) / (smax - smin + 1e-8)

        # (N_words, segment_len)
        vocab_tensor = torch.tensor(self.vocabulary[segment_bit_index], dtype=segment_norm.dtype, device= device)

        distances = torch.abs(vocab_tensor - segment_norm).sum(dim=1)

        best_word_index = torch.argmin(distances).item()

        if not return_index:
            embed_index = self.vocabulary_dist[segment_bit_index] + best_word_index
            return self.word_emb_matrix(embed_index)
        else:
            return segment_bit_index, best_word_index

    def compute_word_embedding(self, X, att_mask, return_index = False):
        """
        Returns the sequence with the corresponding words and their values.
        """
        device = next(self.parameters()).device

        batch_size = X.size(0)

        if not return_index:
            batch_sequence = torch.zeros((batch_size, self.dim_seq, self.dim_model), device = device)
        else:
            batch_sequence = torch.zeros((batch_size, self.dim_seq, 2), device = device)

        for i, x in enumerate(X):
            mask = att_mask[i]
            valid_index = torch.nonzero(mask, as_tuple=True)[0]

            for k in valid_index:
                start = k*self.segment_len
                end = start + self.segment_len
                segment = x[start:end]
                batch_sequence[i, k] = self.get_best_approach(segment, return_index)
    
        return batch_sequence    

    def compute_haar_embedding(self, X, att_mask):
        device = next(self.parameters()).device

        # Move to CPU to copmpute all the coefficients
        X_np = X.detach().cpu().numpy()

        coeffs_list = []
        for i, x in enumerate(X_np):
            # [D1, D2, D3,...] Di = 2*D{i-1}, D1 = 1
            haar_coeffs = pywt.wavedec(x, 'haar')[:self.nhaar_level+1]

            # It is expected that the last Di level has dim_seq elements 
            coeffs_tensor = [torch.from_numpy(hc) for hc in haar_coeffs]

            # We convert the coeffs into a matrix (nhaar_level, dim_seq)
            haar_matrix = [hc.repeat_interleave(self.dim_seq // hc.size(0)) for hc in coeffs_tensor]
            
            haar_matrix = torch.stack(haar_matrix)
            
            # Alternate dimensions to fit in coeffs_batch. Now each segment of each sequence has
            # its corresponding nhaarl_levels coeff 
            coeffs_list.append(haar_matrix.transpose(0, 1))
        
        # Move to device
        coeffs_batch = torch.stack(coeffs_list).to(device=device)

        coeffs_masked = coeffs_batch*att_mask.unsqueeze(-1)
                
        # coeffs_masked (batch_size, dim_seq, nhaar_level)
        # haar_emb_matrix = (nhaar_level, dim_model)
        coeffs_embebed = coeffs_masked @ self.haar_emb_matrix

        return coeffs_embebed


    def get_embedding(self, X):
        assert isinstance(X, torch.Tensor), "Expected X to be a torch.Tensor"

        device = next(self.parameters()).device
        X = X.to(device)

        X_adj, att_mask = self._adjust(X)

        # Get the sequences (segments and correlations)
        sequence_embed = self.compute_word_embedding(X_adj, att_mask)
        
        haar_embed = self.compute_haar_embedding(X_adj, att_mask)

        batchts_embebed = sequence_embed + haar_embed + self.pos_emb_matrix

        return batchts_embebed, att_mask



