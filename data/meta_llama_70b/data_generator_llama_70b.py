from groq import Groq
import json 
from tqdm import tqdm

def complete_chat(messages, model):
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=messages,

        # The language model which will generate the completion.
        model=model,

        # Controls randomness: lowering results in less random completions.
        temperature=0.6,

        # The maximum number of tokens to generate. Requests can use up to
        # 32,768 tokens shared between prompt and completion.
        max_tokens=2024,

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=0.9,

        stop=None,
        # If set, partial message deltas will be sent.
        stream=False,
    )
    return chat_completion.choices[0].message.content

#create a file api_keys.json and add your own key
with open('data/api_keys.json', 'r') as json_file:
    data = json.load(json_file)
    API_KEY = data['groq']


client = Groq(api_key = API_KEY)

file = open('data/meta_llama_70b/prompt.txt', 'r')
prompt = file.read()
file.close() 

messages = messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": "you are a helpful assistant."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": "```json"
        }
    ]

model = "llama3-70b-8192"                   #model to use
n = 1000                                 #number of iterations

out_path = f"./data/mails/g6_{model}.json"
if __name__ == "__main__":
    loop = tqdm(range(0, n))

    out_text = "["
    text = complete_chat(messages, model)
    out_text += (text[2:-5])

    for i in loop:
        text = complete_chat(messages, model)
        out_text += ','
        out_text += (text[2:-5])
        ###checkpointing
        if(i % 5 == 0):
            with open(out_path, 'w') as out_file:
                # Write content to the file
                out_file.write(out_text + ']')

    out_text += ']'
    with open(out_path, 'w') as out_file:
        # Write content to the file
        out_file.write(out_text)
