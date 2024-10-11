from src.RAG.utils.DistilBERT_encoder import DistilBERT_encoder
from src.RAG.utils.retriever import retriever
from groq import Groq
import json 
import torch
import os

from transformers import DistilBertTokenizerFast, DistilBertModel
    

# API call to larger models 
class RAG():
    def __init__(self, api_key, doc_path,  device = 'cpu') -> None:
        
        self.client = Groq(api_key = api_key)
        self.device = device
        self.encoder = DistilBERT_encoder()
        self.encoder.to(device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.current_directory = os.path.dirname(os.path.relpath(__file__))
        self.generator = "llama-3.1-70b-versatile" 
        self.retr = retriever(doc_path)

    def complete_chat(self, messages):
        chat_completion = self.client.chat.completions.create(
            
            messages=messages,
            model=self.generator,
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
        ids = torch.tensor(input['input_ids']).to(self.device)
        mask = torch.tensor(input['attention_mask']).to(self.device)

        token_type_ids = torch.tensor(input['token_type_ids']).to(self.device)
        with torch.no_grad():
            embedding = self.encoder(ids, mask, token_type_ids)
        return embedding

    def reply(self, email):
        query_embedding =  self.BERT_encode(email)
        query_embedding = query_embedding.detach().cpu().numpy()
        texts, dists = self.retr.fetch_topK(query_embedding)   ## fetchin top k responses

        with open(os.path.join(self.current_directory, 'prompts', 'prompt1.txt'), 'r') as file:
            prompt = file.read()

        # Replace the placeholders with the actual values
        prompt = prompt.replace('{email}', email)
        prompt = prompt.replace('{doc1}', texts[0][0])
        prompt = prompt.replace('{doc2}', texts[0][1])
        prompt = prompt.replace('{doc3}', texts[0][2])
        
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
        reply =  self.complete_chat(messages)
        return reply


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('api_keys.json', 'r') as json_file:
        data = json.load(json_file)
        api_key = data['groq']

    path = './data/docs/encodings.json'

    text = "What is he prerquisite for Deep Learning course?"    
    
    r = RAG(api_key, doc_path=path,  device=device)
    reply = r.reply(text)

    print(reply)

    
