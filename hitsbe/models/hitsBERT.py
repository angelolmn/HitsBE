import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

from hitsbe.hitsbe import Hitsbe

class HitsBERTClassifier(nn.Module):
    def __init__(self, batch_size, sequence_length=128, num_classes=3, bert_model_name='bert-base-uncased'):
        super(HitsBERTClassifier, self).__init__()
        # Load BERT for sequence classification with hidden states output
        self.bert = BertForSequenceClassification.from_pretrained(
            bert_model_name, output_hidden_states=True
        )

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls_token = self.tokenizer.cls_token  
        cls_token_id = self.tokenizer.convert_tokens_to_ids(cls_token)
        with torch.no_grad():
            self.cls_embedding = self.bert.bert.embeddings.word_embeddings(
                torch.tensor([cls_token_id])
            )
        
        # Instantiate the Hitsbe module
        self.hitsbe = Hitsbe()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Classification layer: the [CLS] vector (first position) is used to predict the class
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, data, attention_mask=None):
        # Obtain the HitsBE embeddings
        hitsbe_embeddings = self.hitsbe.get_embedding(data) # (batch_size, sequence_length, hidden_size)
        
        # Prepend [CLS] token to each sequence
        cls = self.cls_embedding.to(hitsbe_embeddings.device)  # (1, hidden_size)
        # unsqueeze(0): (1,1,hidden_size)
        # expand(hitssebe_embedding.size(0), -1, -1): (batch_size, 1, hidden_size)
        cls = cls.unsqueeze(0).expand(hitsbe_embeddings.size(0), -1, -1)  
        # Concatenate cls and hitsbe_embedding on the dimension 1
        embeddings_with_cls = torch.cat((cls, hitsbe_embeddings), dim=1)  # (batch_size, sequence_length + 1, hidden_size)
        
        # Adjust the attention mask
        if attention_mask is None:
            internal_mask = self.hitsbe.att_mask.to(embeddings_with_cls.device) # (seq_length)
            # Concatenate ones (batch_size,1) with (1, seq,length + 1) ot dim 1, (batch_size, se_length + 1)
            full_mask = torch.cat((torch.ones(embeddings_with_cls.size(0), 1, device=embeddings_with_cls.device),
                                    internal_mask.unsqueeze(0).expand(embeddings_with_cls.size(0), -1)), dim=1)
            attention_mask = full_mask
        else:
            attention_mask = attention_mask.to(embeddings_with_cls.device)
        
        outputs = self.bert(inputs_embeds=embeddings_with_cls, attention_mask=attention_mask)
        # Get the last hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch_size, sequence_length + 1, hidden_size)
        # Extract the [CLS] vector 
        cls_vectors = last_hidden[:, 0, :]  # (batch_size, hidden_size)
        # Calculate the classification logits
        logits = self.classifier(cls_vectors)  # (batch_size, num_classes)
        
        return logits
