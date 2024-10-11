import warnings
warnings.filterwarnings("ignore")
import torch 
import argparse
import json
from src.RAG.rag import RAG
from src.classifier.DistillBERT import DistilBERTClass
from transformers import DistilBertTokenizerFast

parser = argparse.ArgumentParser(description="Automatic email classification and reply")
parser.add_argument('input_file', type=str, help='The input file to process')
parser.add_argument('--output_file', type=str, default='output.txt', help='The output file name (default: output.txt)')
parser.add_argument('--doc_path', type=str, default='./data/docs/encodings.json', help='document path for retrieval')
parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('api_keys.json', 'r') as json_file:
    data = json.load(json_file)
    api_key = data['groq']

def classify(text, classifier, tokenizer):
    input = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation = True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True
    )

    ids = torch.tensor(input['input_ids']).to(device)
    mask = torch.tensor(input['attention_mask']).to(device)
    token_type_ids = torch.tensor(input['token_type_ids']).to(device)

    classifier.eval()

    with torch.no_grad():
        probs = classifier(ids, mask, token_type_ids)
        probs = torch.nn.functional.softmax(probs)

    return probs

if __name__ == "__main__":

    with open(args.input_file, 'r') as file:
        text = file.read()

    classifier = DistilBERTClass()
    classifier.load_state_dict(torch.load('weights/DistillBERT.pt'))
    classifier.to(device)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    probs = classify(text, classifier, tokenizer)
    label = torch.argmax(probs).item()
    print(label)

    if(label == 0):
        print("sensitive email, will be replied only by proffesor")

    elif(label == 1):
        print("General email")
        r = RAG(api_key, doc_path=args.doc_path,  device=device)
        reply = r.reply(text)
        
        if args.verbose:
            print("Reply: \n")
            print(reply)
        
        with open(args.output_file, 'w') as file:
            file.write(reply)

    else:
        print("Reaserch Query")
        r = RAG(api_key, doc_path=args.doc_path,  device=device, retrieval_cutoff=2)  #retrieval_cutoff is lower because research documents should be concise to field of research 
        reply = r.reply(text)
        
        if args.verbose:
            print("The followng is the draft and must be aproved by the proffesor: \n")
            print(reply)
        
        with open(args.output_file, 'w') as file:
            file.write(reply)