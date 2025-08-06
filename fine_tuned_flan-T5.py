from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, GenerationConfig
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
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


def get_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return trainable_model_params, all_model_params


trainable_params, total_params = get_number_of_trainable_model_parameters(original_model)
print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {total_params}")

# --------------------------------------------------------
# 5. Tokenize function (fixed: no return_tensors)
# --------------------------------------------------------



def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]

    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids  # do not put it to device eg to(device). it leads to catastripic forgetting. 
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary', ])

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

# Data collator for padding and label masking
data_collator = DataCollatorForSeq2Seq(tokenizer, model=original_model)

# --------------------------------------------------------
# 6. Training setup
# --------------------------------------------------------
output_dir = './dialogue-summary-full-training-checkpoint'

# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=1e-3,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     logging_steps=1,
#     # evaluation_strategy="steps",  # Ensure validation runs
#     eval_steps=1,
#     max_steps=1,  # For quick testing  ## this is what being plotted in the graph  
# )

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    num_train_epochs=200,
    weight_decay=0.01,
    logging_steps=5,
    max_steps= 500,  # For quick testing
    evaluation_strategy = "steps",  # Ensure validation runs
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator
)

# --------------------------------------------------------
# 7. Train model
# --------------------------------------------------------
start_time = time.time()
trainer.train()
end_time = time.time()
training_duration = end_time - start_time  # seconds
print(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes).")



# Save these training stats
training_stats = {
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "training_time_seconds": training_duration,
    "training_time_minutes": training_duration / 60,
    "num_train_epochs": training_args.num_train_epochs,
    "max_steps": training_args.max_steps,
    "learning_rate": training_args.learning_rate,
}
# Save as JSON for easy later comparison
os.makedirs("train_stats", exist_ok=True)

with open("train_stats/full_fine_tuning_training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=4)

print("Training stats saved to training_stats.json")



#### TO save the models and tokenizers ####

fine_tuned_model_path="./dialogue-summary-full-training-checkpoint-local"

original_model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

# --------------------------------------------------------
# 8. Plot training vs validation loss (using built-in logs)
# --------------------------------------------------------
train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
val_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
train_steps = [log["step"] for log in trainer.state.log_history if "loss" in log]
val_steps = [log["step"] for log in trainer.state.log_history if "eval_loss" in log]



plt.figure()
plt.plot(train_steps[8:], train_loss[8:], label="Training Loss")
plt.plot(val_steps[8:], val_loss[8:], label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/training_vs_validation_loss.png")
print("Loss plot saved as training_vs_validation_loss.png")


# as the original model as been full trained
instruction_fine_tuned_model = original_model

# --------------------------------------------------------
# 9. Test inference
# --------------------------------------------------------
index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summaries = dataset['test'][index]['summary']
print(dash_line)
print(f"HUMAN BASELINE SUMMARY:\n{human_baseline_summaries}")
print(dash_line)

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # bring it to device to keep the tensor in the same device as the model



instruct_model_outputs = instruction_fine_tuned_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summaries}')


print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')


##################################  Lets evaluate the mode now and save it as csv #################### 
rouge = evaluate.load('rouge')

# Select a subset of dialogues from the test set to generate summaries.
dialogues = dataset['test']['dialogue']
human_baseline_summaries = dataset['test']['summary']

instruction_fine_tuned_model_summaries = []

for dialogue in dialogues:
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = instruction_fine_tuned_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    instruction_fine_tuned_model_summaries.append(summary_text)

# Save only the trained model's summaries as a CSV file.
zipped_summaries = list(zip(human_baseline_summaries,  instruction_fine_tuned_model_summaries))

df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries',  'instruct_model_summaries'])
print(dash_line)
print("printing DF")
print(df)
print(dash_line)
output_dir = "./model_outputs"
os.makedirs(output_dir, exist_ok=True)
csv_filepath = os.path.join(output_dir, "instruction_fine_tuned_model_summaries.csv")
df.to_csv(csv_filepath, index=False)
print(f"CSV file saved as {csv_filepath}")
