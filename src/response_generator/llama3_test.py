from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

token = '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'

login(token = token)

tokenizer = AutoTokenizer.from_pretrained(
"meta-llama/Meta-Llama-3-8B",
cache_dir="/tmp"
)

model = AutoModelForCausalLM.from_pretrained(
"meta-llama/Meta-Llama-3-8B",
cache_dir="/tmp",
device_map="auto",
)