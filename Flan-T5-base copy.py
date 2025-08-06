from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# Load dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
print(dataset)

model_name = 'google/flan-t5-base'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load base model and tokenizer
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    # Tokenize inputs and labels, returning tensors for Trainer usage
    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt")
    example['input_ids'] = inputs.input_ids
    example['attention_mask'] = inputs.attention_mask
    example['labels'] = labels.input_ids
    return example

# Map tokenize function over all splits
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])

print("---------------------------------- Before limiting the dataset -----------------------------------------------")
print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")
print("---------------------------------------------------------------------------------")

# Filter dataset to reduce size (keep every 100th example)
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 10 == 0, with_indices=True)

print("---------------------------------------------------------------------------------")
print(f"Shapes of the datasets after filtering:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")
print("---------------------------------------------------------------------------------")

output_dir = f'./dialogue-summary-training-{int(time.time())}'

# Configure LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply LoRA to base model
peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

# Training args
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-4,
    num_train_epochs=500,
    logging_steps=1,
    max_steps=500,
    report_to='none'  # no logging service
)

# Trainer
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets['train']
)

peft_trainer.train()

# Save the PEFT model
peft_model_path = './peft-dialogue-summary-checkpoint-local'
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# ------------------------
# Inference on test set
# ------------------------

# Reload base model and tokenizer for inference
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load LoRA weights on base model properly
peft_model = PeftModel.from_pretrained(peft_model_base, peft_model_path, torch_dtype=torch.bfloat16).to(device)
peft_model.eval()
original_model.eval()

index = 200  # random test sample index
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

original_model_outputs = original_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(f'Human Baseline summary: \n{human_baseline_summary}\n')
print(f'Original Model Output \n{original_model_text_output}\n')
print(f'PEFT Model Output \n{peft_model_text_output}\n')

# ------------------------
# Evaluate model summaries on first 10 test examples
# ------------------------

human_baseline_summaries = dataset['test'][0:10]['summary']
original_model_summaries = []
peft_model_summaries = []

for dialogue in dataset['test'][0:10]['dialogue']:
    prompt = f"""
    Summarize the following conversation. 

    {dialogue}

    Summary: """
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    peft_model_summaries.append(peft_model_text_output)

# Create DataFrame with results
zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
print("---------------------------------------------------------------------------------")
print(df)
print("---------------------------------------------------------------------------------")

# Compute ROUGE scores
rouge = evaluate.load('rouge')

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True
)

print(f'Original Model ROUGE: \n{original_model_results}\n')
print(f'PEFT Model ROUGE: \n{peft_model_results}\n')
