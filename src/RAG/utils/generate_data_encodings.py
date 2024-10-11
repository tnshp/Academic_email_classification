from DistilBERT_encoder import DistilBERT_encoder
from transformers import DistilBertTokenizerFast
import torch
import os
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = DistilBERT_encoder()
model.to(device)

data_to_save = []
if __name__ == "__main__":
    path = './data/docs'

    file_list = os.listdir(os.path.join(path, 'docs'))
    for f in file_list:
        with open(os.path.join(path, 'docs', f), 'r') as doc:
            text = doc.read()

        inputs = tokenizer.encode_plus(
            text,
            None,
            max_length=512,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        
        ids = torch.tensor(inputs['input_ids']).to(device)
        mask = torch.tensor(inputs['attention_mask']).to(device)
        token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)

        outputs = model(ids, mask, token_type_ids)

        outputs = outputs.detach().cpu().numpy()

        data_to_save.append({'path':  os.path.join(path, 'docs', f), 'embedding':outputs.tolist()})
    
       
    json_file_path = os.path.join(path, 'encodings.json')
    with open(json_file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)

