#!/usr/bin/env python
# coding: utf-8

# # Block and Breach - Qwen / Direct Pass / Baseline - Response Generation
# 1. **Target Model:** ```Qwen/Qwen2.5-7B-Instruct```
# 2. **Attack Method:** ```Direct Pass``` (Default)
# 3. **Defense Method:** ```Baseline``` (Default)

# > **Note on Evaluation (Judgment):**  
# > This notebook only generates responses from the target model and saves them (along with attention‑based xAI scores) to CSV and JSON files.  
# > **Judgment of response correctness/safety is performed offline** using external state‑of‑the‑art judge models (e.g., DeepSeek, Gemini, GPT‑4, or Llama‑3.3‑70B).  
# > The evaluation pipeline is therefore decoupled: generation → saving → external judgment.

# ## 0. Debug Switch

# In[1]:


DEBUG = False


# ## 1. Setup, Imports & Configuration

# In[2]:


# Requirements
get_ipython().system('pip install -U bitsandbytes>=0.46.1')


# In[3]:


# Libraries
import os
import gc
import time
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration & Constants
TARGET_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "/kaggle/input/datasets/nidalshahin/x-guard-bench/eval_500.csv"

# Attack & defense naming
ATTACK_METHOD = "Direct-Pass"
DEFENSE_METHOD = "Baseline"

# DataFrame display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[4]:


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")

os.environ["HF_TOKEN"] = secret_value_0


# ## 2. Target Model Logic

# In[5]:


class TargetLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.short_name = model_name.split('/')[-1]
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config,
            device_map="auto",
            output_attentions=True, # Required for attention-based xAI
        )
        self.chat_template = self.tokenizer.apply_chat_template # Chat template for Instruct models

    def generate_and_get_xai(self, prompt, max_new_tokens=128):
        # Format prompt as a user message (for instruct models)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_attentions=True,
                do_sample=False,   # deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response_ids = outputs.sequences[0][input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Attention extraction
        attention_scores = {}
        if outputs.attentions:
            # Extract attention from the first generation step (input processing)
            # Tuple: (layer) -> [batch, head, seq, seq]
            first_step_attentions = outputs.attentions[0] 
            avg_attention_layers = torch.stack(first_step_attentions).mean(dim=0) 
            avg_attention = avg_attention_layers.mean(dim=(0, 1))

            # Sum columns to find which input tokens received most attention
            token_influence = avg_attention.sum(dim=0)[:input_length].to(torch.float32).cpu().numpy()
            
            if token_influence.sum() > 0:
                token_influence = token_influence / token_influence.sum()
                
            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            for token, score in zip(input_tokens, token_influence):
                clean_token = token.replace('Ġ', ' ').replace(' ', ' ')
                attention_scores[clean_token] = float(score.item()) if hasattr(score, 'item') else float(score)
        
        return response_text, attention_scores


# ## 3. Main Execution Block

# In[6]:


# Load prompts data
df_full = pd.read_csv(DATA_PATH)

if DEBUG:
    df_eval = df_full.head(2)  
else:
    df_eval = df_full

print("Succesfully loaded the data...\n")


# In[7]:


# Initialize Components
llm = TargetLLM(TARGET_MODEL_ID)

results = []
per_prompt_attentions = []   # list for per-prompt attention data
all_token_scores = {}        # for aggregated JSON


# In[8]:


# Run Evaluation
print(f"Starting evaluation on {len(df_eval)} prompts...")

for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
    prompt_text = row['prompt']
    label = row['label']
    category = row['category']
    intent = row['intent']
    source = row['source']
    
    # Generate response and capture raw attention (for xAI)
    response, attention = llm.generate_and_get_xai(prompt_text)
    
    # Store results
    results.append({
        "prompt": prompt_text,
        "category": category,
        "label": label,
        "intent": intent,
        "source": source,
        "target_model": llm.short_name,
        "attack_method": ATTACK_METHOD,
        "defense_method": DEFENSE_METHOD,
        "model_response": response
    })
    
    # Store per-prompt attention for JSON
    per_prompt_attentions.append({
        "prompt": prompt_text,
        "attention_scores": attention
    })

     # Aggregated attention (for global charts)
    for token, score in attention.items():
        if token not in all_token_scores:
            all_token_scores[token] = []
        all_token_scores[token].append(score)


# ## 4. Saving

# In[9]:


# Convert to DataFrame and save CSV
results_df = pd.DataFrame(results)
csv_output_path = f"/kaggle/working/{llm.short_name}_{ATTACK_METHOD}_{DEFENSE_METHOD}_responses.csv"
results_df.to_csv(csv_output_path, index=False)
print(f"Saved results CSV to {csv_output_path}")


# In[10]:


# Save per-prompt attention scores as JSON
with open(f"/kaggle/working/{llm.short_name}_{ATTACK_METHOD}_{DEFENSE_METHOD}_xai-per-prompt.json", "w") as f:
    json.dump(per_prompt_attentions, f, indent=2)
print("Saved per-prompt attention JSON")


# In[11]:


# Save aggregated attention scores JSON
with open(f"/kaggle/working/{llm.short_name}_{ATTACK_METHOD}_{DEFENSE_METHOD}_xai-aggregated.json", "w") as f:
    json.dump(all_token_scores, f, indent=2)
print("Saved aggregated attention JSON")


# In[12]:


# Preview first few rows
print("\nFirst 2 rows of saved results:")
results_df.head(2)

