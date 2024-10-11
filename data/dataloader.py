import torch 
import os 
import json 


label_to_logit = {
    "Sensitive": 0,
    "Sensitive Email": 0,
    "General Query": 1,
    "Research Queries": 2
}

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len= 128, num_classes = 3):
        file_list = os.listdir(path)
        data = []
        for f in file_list:
            with open(os.path.join(path, f), 'r') as json_file:
                data += json.load(json_file)
    
        self.labels = [item['label'] for item in data]
        self.emails = [item['email'] for item in data]

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
        self.encodings = self.tokenizer(self.emails, truncation=True, padding=True)

    def __getitem__(self, idx):
        text = self.emails[idx]
        # text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[self.labels[idx]] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': target
        }

    def __len__(self):
        return len(self.labels)
    
if __name__ == '__main__':
    from transformers import DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset = EmailDataset('data/mails', tokenizer)
    # print(dataset[0]['targets'])
    print(len(dataset))