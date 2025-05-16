import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from hitsbe import Hitsbe, HitsbeConfig

import os

class HitsBERT(nn.Module):
    def __init__(self, bert_config = None, hitsbe= None):
        super().__init__()
        # Load BERT model via BertConfig     
        # El comentario de abajo deberia estar en la memoria o en el README como tutorial de uso
        # If we want to use "bert-base-uncased" or "bert-large-uncased"
        # we must define the BertConfig externally using `BertConfig.from_pretrained(...)`
        # and load pretrained weights separately using `BertModel.from_pretrained(...).state_dict()`
        # Example: 
        # bert = BertModel.from_pretrained("bert-base-uncased")
        # config = bert.config
        # model = HitsBERT(bert_config=config)
        # model.bert.load_state_dict(bert.state_dict())
        self.config = bert_config or BertConfig()
        self.bert = BertModel(self.config)
        
        # Instantiate the Hitsbe module.
        self.hitsbe = hitsbe or Hitsbe()
        
        # Initialize the classifier token [CLS]. It must be trainable
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))
        nn.init.normal_(self.cls_embedding, mean=0.0, std=self.config.initializer_range)
    
    # Prepend [CLS] token to each sequence
    # embeds (batch_size, dim_seq, hidden_size)
    # att_mask (batch_size, dim_seq) 
    def concat_cls(self, embeds, att_mask):
        batch_size = embeds.size(0)

        # expand(batch_size, -1, -1): (batch_size, 1, hidden_size)
        cls = self.cls_embedding.expand(batch_size, -1, -1)
        # Concatenate cls and hitsbe_embedding on the dimension 1
        embeddings_with_cls = torch.cat((cls, embeds), dim=1)  # (batch_size, dim_seq + 1, hidden_size)
        
        cls_mask = torch.ones((batch_size, 1), dtype=att_mask.dtype, device=att_mask.device)
        # (batch_size, dim_seq + 1) 
        att_mask_cls =  torch.cat((cls_mask, att_mask), dim=1)
        
        return embeddings_with_cls, att_mask_cls

    def forward(self, data):        
        # Obtain the HitsBE embeddings
        hitsbe_embeddings, att_mask = self.hitsbe.get_embedding(data) # (batch_size, dim_seq, hidden_size)
                
        embeddings_cls, att_mask_cls = self.concat_cls(hitsbe_embeddings, att_mask)
                        
        return self.bert(inputs_embeds=embeddings_cls, attention_mask=att_mask_cls)
    
    def save_pretrained(self, path, filename = "hitsbert_model.bin"):
        os.makedirs(path, exist_ok =True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(path, filename))
        # Save BERT config
        self.config.to_json_file(os.path.join(path, "config_BERT.json"))
        # Save hitsbe 
        self.hitsbe.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, filename = "hitsbert_model.bin"):
        bert_config = BertConfig.from_json_file(os.path.join(path, "config_BERT.json"))
        hitsbe_config = HitsbeConfig.from_pretrained(path)

        hitsbe = Hitsbe(hitsbe_config)
        model = cls(bert_config=bert_config, hitsbe=hitsbe)

        # map_location load "cpu" if the saved default is not available
        model.load_state_dict(torch.load(os.path.join(path, filename), map_location="cpu"))
        return model
    

class HitsBERTPretraining(nn.Module):
    def __init__(self, model: HitsBERT):
        super().__init__()

        self.model = model

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.model.config.hidden_size))
        # Weights initialized with 0 mean and initializer_range std
        nn.init.normal_(self.mask_token, mean=0.0, std=self.model.config.initializer_range)

    # With this algorithm we can compute the option embeddings 
    # and the time series embeddings only once per step.
    # X: time series, 3 for each test (batch_size, 3, ts_len)
    # MLM: tensor torch binario con 1 en [MASK] (batch_size, dim_seq)
    # solutions: (batch_size,)
    def forward(self, X, MLM, solutions):
        device = next(self.parameters()).device

        bsz, noptions, ts_len = X.shape
        dim_seq = MLM.size(1)
        dim_model = self.model.hitsbe.dim_model

        # --------------------------------------
        # (1) GET EMBEDDINGS 
        # --------------------------------------

        # X dimension: (batch_size, 3, ts_len) -> (batch_size*3, ts_len)
        X_reshaped = X.reshape(-1, ts_len)

        # (batch_size*3, dim_seq, dim_model)
        X_embed, att_mask = self.model.hitsbe.get_embedding(X_reshaped)

        # --------------------------------------
        # (2) MASK INPUTS 
        # --------------------------------------

        # embeddings dimension: (batch_size*3, dim_seq, dim_model) -> 
        # -> (batch_size, 3, dim_seq, dim_model)
        embed_grouped = X_embed.reshape(bsz, noptions, dim_seq, dim_model)

        batch_index = torch.arange(bsz)
        
        # Select the ts which will be the solution so must be introduced at BERT
        # (batch_size, dim_seq, dim_model) 
        embed_to_BERT = embed_grouped[batch_index, solutions]

        # No hace falta usar solutions pero ya que lo tenemos lo aprovechamos
        # Es la misma mascara para las n_options opciones
        att_mask = att_mask[batch_index, solutions]

        # (batch_size, 1, dim_seq, dim_model) -> (batch_size, dim_seq, dim_model) 

        # Matriz con el token [MASK] para facilitar enmascaramiento
        # (batch_size, dim_seq, dim_model)
        masktoken_matrix = self.mask_token.expand(bsz, dim_seq, -1)

        # (batch_size, dim_seq, 1)
        MLM_comp = 1 - MLM

        # Replace the corresponding mask index by [MASK] token
        # (batch_size, dim_seq, dim_model)
        embeddings_masked = embed_to_BERT*MLM_comp.unsqueeze(2) + masktoken_matrix*MLM.unsqueeze(2)
        
        # (batch_size, dim_seq + 1(CLS), dim_model)
        embeddings_cls, attention_mask_cls = self.model.concat_cls(embeddings_masked, att_mask)

        # --------------------------------------
        # BERT 
        # --------------------------------------

        # (batch_size, dim_seq + 1(CLS), dim_model)                
        output = self.model.bert(inputs_embeds=embeddings_cls, attention_mask=attention_mask_cls)

        # --------------------------------------
        # (3) COMPUTE LOGITS 
        # --------------------------------------

        # Remove [CLS] token 
        # (batch_size, dim_seq + 1(CLS), dim_model) -> (batch_size, dim_seq, dim_model)
        last_hidden_state = output.last_hidden_state
        answers = last_hidden_state[:, 1:, :]

        # answers dimension: (batch_size, dim_seq, dim_model);             
        # embed_grouped dimension: (batch_size, num_options, dim_seq, dim_model)
        # scalar product of the words 
        # <(batch_size, dim_seq, dim_model),(batch_size, num_options, dim_seq, dim_model)> =
        # = (batch_size, num_options, dim_seq)
        token_scores = torch.einsum('bij,bkij->bki', answers, embed_grouped)

        # Sum the results of the scalar product in each option to get the logits
        # (batch_size, num_options, dim_seq) -> (batch_size, num_options, 1)
        logits = token_scores.sum(dim = 2)

        return logits # Each row contains the logits for each task
    
    def save_pretrained(self, path, filename = "hitsbertpretrain_model.bin"):
        os.makedirs(path, exist_ok =True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(path, filename))
        # Save BERT config
        self.config.to_json_file(os.path.join(path, "config_BERT.json"))
        # Save hitsbe 
        self.hitsbe.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, filename = "hitsbertpretrain_model.bin"):
        bert_config = BertConfig.from_json_file(os.path.join(path, "config_BERT.json"))
        hitsbe_config = HitsbeConfig.from_pretrained(path)

        hitsbe = Hitsbe(hitsbe_config)
        model = cls(bert_config=bert_config, hitsbe=hitsbe)

        # map_location load "cpu" if the saved default is not available
        model.load_state_dict(torch.load(os.path.join(path, filename), map_location="cpu"))
        return model
    
# REVISAR BIEN
class HitsBERTClassifier(nn.Module):
    def __init__(self, model: HitsBERT, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.model.config.initializer_range)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, X):
        # X: (batch_size, ts_len)
        hitsbe_embeddings, att_mask = self.model.hitsbe.get_embedding(X)
        embeddings_cls, att_mask_cls = self.model.concat_cls(hitsbe_embeddings, att_mask)

        output = self.model.bert(inputs_embeds=embeddings_cls, attention_mask=att_mask_cls)
        cls_embedding = output.last_hidden_state[:, 0]  # (batch_size, hidden_size)

        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)
        return logits

    def save_pretrained(self, path, filename="hitsbert_classifier.bin"):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
        }, os.path.join(path, filename))
        self.model.config.to_json_file(os.path.join(path, "config.json"))

    @classmethod
    def from_pretrained(cls, path, filename="hitsbert_classifier.bin", hitsbe=None):
        config = BertConfig.from_json_file(os.path.join(path, "config.json"))
        hitsbert = HitsBERT(bert_config=config, hitsbe=hitsbe)

        checkpoint = torch.load(os.path.join(path, filename), map_location="cpu")
        classifier = cls(hitsbert, num_classes=checkpoint["num_classes"])
        classifier.load_state_dict(checkpoint["state_dict"])
        return classifier
