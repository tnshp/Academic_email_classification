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
    def __init__(self, path, tokenizer):
        file_list = os.listdir(path)
        data = []
        for f in file_list:
            with open(os.path.join(path, f), 'r') as json_file:
                data += json.load(json_file)
    
        self.labels = [item['label'] for item in data]
        self.emails = [item['email'] for item in data]

        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(self.emails, truncation=True, padding=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
if __name__ == '__main__':
    dataset = EmailDataset('data/mails')
    print(dataset[0])