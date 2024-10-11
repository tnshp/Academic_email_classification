from utils.DistilBERT_encoder import DistilBERT_encoder
from utils.retriever import retriever
from groq import Groq
import json 
import torch

from transformers import DistilBertTokenizerFast, DistilBertModel
    


with open('api_keys.json', 'r') as json_file:
    data = json.load(json_file)
    API_KEY = data['groq']

client = Groq(api_key = API_KEY)

# API call to larger models 
class RAG():
    def __init__(self, api_key, doc_path,  device = 'cpu') -> None:
        
        self.client = Groq(api_key = api_key)

        self.encoder = DistilBERT_encoder()
        self.encoder.to(device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.retr = retriever(doc_path)

    def complete_chat(self, messages, model):
        chat_completion = client.chat.completions.create(
            
            messages=messages,
            model=model,
            temperature=0.6,
            max_tokens=2024,
            top_p=0.9,

            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    
    ### embedding
    def BERT_encode(self, text):
        input = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = torch.tensor(input['input_ids']).to(device)
        mask = torch.tensor(input['attention_mask']).to(device)

        token_type_ids = torch.tensor(input['token_type_ids']).to(device)
        with torch.no_grad():
            embedding = self.encoder(ids, mask, token_type_ids)
        return embedding

    def reply(self, text):
        query_embedding =  self.BERT_encode(text)
        query_embedding = query_embedding.detach().cpu().numpy()
        texts, dists = self.retr.fetch_topK(query_embedding)   ## fetchin top k responses
        
        
        messages =[
        {
            "role": "system",
            "content": "you are a helpful assistant. replying to emails"
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]
        return texts, dists


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('api_keys.json', 'r') as json_file:
        data = json.load(json_file)
        api_key = data['groq']

    path = './data/docs/encodings.json'

    text = "What is he prerquisite for Deep Learning course?"    
    
    r = RAG(api_key, doc_path=path,  device=device)
    replies, dists = r.reply(text)

    print(dists)

    
