# Automated email classification and response system 
The following system aims to automate emails reply for HOD of department in any Academic instititue 

I have used Distilled BERT as a classifier for emails into 3 catgories:

- **Sensitive Emails**:
Emails related to corporate enquiries or sensitive information (confidential partnerships or
legal matters) are escalated to the respective HODs for manual response.
- **General Information**:
For non-sensitive emails that fall under public knowledge (academic schedules, research
access, department procedures), the system drafts a response by pulling data from the
university’s document repositories.
- **Research Queries**:
If the system identifies an email related to shared research data, it verifies whether the
required information is available in the database. If so, it formulates an appropriate response
and replies

The Distilled BERT was used as a classifier. The training was done on emails generated from Larger models ( Llama3.1 70b, Llama3.1 70b_versattile, mixtral). This emails were generated for each of three categories and training was carried out on collab T4 GPUs(refer to [distill_bert_train.py]).

Following a RAG system is implemented for generatiing replies. the RAG encoder used is Distill BERT base model used for vector embeddings for Documents and query. Following a KNN search top 3 documents below the distance threshold are fed to generator. llama3 70b from Groq API was used a seq2seq generator after retrieval


## Instruction

You will need a groq API key to run the model. It should stored under filename api_keys.json in format {'groq': [API_KEY]}

download the trained weights from: [https://iitgnacin-my.sharepoint.com/:u:/g/personal/21110221_iitgn_ac_in/EbvDVsuHY9JChQqhEHmR43AB_HWQZsEVvNLV40vj3ur65w?e=ucnENo](link) and save in weights folder


Then run 
```
python app.py [input_file_path] --output_file [Output_fle_path]
```
from example
```
python app.py examples/Q1_General.txt --output_file examples/A1.txt
```

You can also add your own documents in data/docs/docs 
and run src/RAG/utils/generate_data_encodings.py to generate vector embeddings for the documents



