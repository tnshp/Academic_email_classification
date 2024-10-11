# Use a pipeline as a high-level helper
import torch  
from transformers import DistilBertModel

class DistilBERT_encoder(torch.nn.Module):
    def __init__(self):
        super(DistilBERT_encoder, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output= pooler
        return output

