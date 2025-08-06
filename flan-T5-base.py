from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import os

# --------------------------------------------------------
# 1. Set device (GPU if available, otherwise CPU)
# --------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dash_line = '-'.join('' for x in range(100))
print()
print()
print(dash_line)
print(f"Using device: {device}")
# Print GPU and system resource details
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")   
else:
    print("CUDA is not available. Using CPU only.")

print()
print(dash_line)
print()




# --------------------------------------------------------
# 2. Load dataset
# --------------------------------------------------------

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

print(dataset)

# --------------------------------------------------------
# 3. Load model and tokenizer
# --------------------------------------------------------
model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------
# 4. Check trainable parameters
# --------------------------------------------------------


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))




####################  Lets Perform Zero shot to see if the model is already good enough without fine-tuning ####################


index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt').to(device) 
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"], 
        max_new_tokens=200,
    )[0], 
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')




##################################  Lets evaluate the mode now and save it as csv #################### 
rouge = evaluate.load('rouge')

# Select a subset of dialogues from the test set to generate summaries.
dialogues = dataset['test']['dialogue']
human_baseline_summaries = dataset['test']['summary']

original_model_summaries = []

for dialogue in dialogues:
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) 
    outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    original_model_summaries.append(summary_text)



# Save only the trained model's summaries as a CSV file.
zipped_summaries = list(zip(human_baseline_summaries,  original_model_summaries))

df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries',  'original_model_summaries'])
print(dash_line)
print("printing DF")
print(df)
print(dash_line)
output_dir = "./model_outputs"
os.makedirs(output_dir, exist_ok=True)
csv_filepath = os.path.join(output_dir, "original_model_summaries.csv")

df.to_csv(csv_filepath, index=False)
print("CSV file saved as original_model_summaries.csv")





