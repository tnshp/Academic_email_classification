# Automated email classification and response system 
The following system aims to automate emails reply for HOD of department in any Academic instititue 

I have used Distilled BERT as a classifier for emails into 3 catgories:

- Sensitive Emails:
■ Emails related to corporate enquiries or sensitive information (confidential partnerships or
legal matters) are escalated to the respective HODs for manual response.
- General Information:
For non-sensitive emails that fall under public knowledge (academic schedules, research
access, department procedures), the system drafts a response by pulling data from the
university’s document repositories.
- Research Queries:
If the system identifies an email related to shared research data, it verifies whether the
required information is available in the database. If so, it formulates an appropriate response
and replies

For all the non sensitive emails, the system drafts a reply. I have used RAG model to retrieve the required documents which are then fed to seq2seq model for generating reply. I have used a llama3 70b hosted on groq model for generation task.   

The BERT classifier was trained on google colab t4 gpu. I  have generated the email data for each category using opensource LLMs, it can be found data folder/llama_70b, with aprox ~ 9000 emails from all the 3 categories. 
```
python app.py [input_files] --output_file 
```


I have created 
download weights: [https://iitgnacin-my.sharepoint.com/:u:/g/personal/21110221_iitgn_ac_in/EbvDVsuHY9JChQqhEHmR43AB_HWQZsEVvNLV40vj3ur65w?e=ucnENo] and save in weights folder




