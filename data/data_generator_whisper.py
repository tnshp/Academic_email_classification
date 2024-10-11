from groq import Groq
import json
from tqdm import tqdm

from openai import OpenAI

with open('data/api_keys.json', 'r') as json_file:
    data = json.load(json_file)
    API_KEY = data['open_router']

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)
response = client.chat.completions.with_raw_response.create(
    messages=[{
        "role": "user",
        "content": "Say this is a test",
    }],
    model="whisper-large-v3",
)
print(response.headers.get('x-ratelimit-limit-tokens'))
# get the object that `chat.completions.create()` would have returned
completion = response.parse()
print(completion)