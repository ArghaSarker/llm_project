import os
import json
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) LOAD CSVs & COMPUTE METRICS
# -------------------------------------------------------------------
original_model    = pd.read_csv("model_outputs/original_model_summaries.csv")
fine_tuned_model  = pd.read_csv("model_outputs/instruction_fine_tuned_model_summaries.csv")
peft_model        = pd.read_csv("model_outputs/peft_model_summaries.csv")

rouge = evaluate.load('rouge')
bleu  = evaluate.load('bleu')

def compute_metrics(df, pred_col):
    refs = df['human_baseline_summaries'].tolist()
    preds = df[pred_col].tolist()
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    b = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return r, b

orig_r, orig_b = compute_metrics(original_model,    'original_model_summaries')
ft_r,   ft_b   = compute_metrics(fine_tuned_model,  'instruct_model_summaries')
peft_r, peft_b = compute_metrics(peft_model,        'peft_model_summaries')

scores = {
    'ROUGE-1': [orig_r['rouge1'], ft_r['rouge1'], peft_r['rouge1']],
    'ROUGE-2': [orig_r['rouge2'], ft_r['rouge2'], peft_r['rouge2']],
    'ROUGE-L': [orig_r['rougeL'], ft_r['rougeL'], peft_r['rougeL']],
    'BLEU':    [orig_b['bleu'],   ft_b['bleu'],   peft_b['bleu']],
}

models = ['Original', 'Fine-tuned', 'PEFT']
x = np.arange(len(scores))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
rects = []
for i, model in enumerate(models):
    rects.append(
        ax.bar(x + (i-1)*width, [scores[k][i] for k in scores], width, label=model)
    )

ax.set_xticks(x)
ax.set_xticklabels(list(scores.keys()))
ax.set_ylabel("Score")
ax.set_title("Model Comparison by Metric")
ax.legend()

def label_bars(bar_container):
    for bar in bar_container:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for rc in rects:
    label_bars(rc)

plt.tight_layout()
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/grouped_by_score.png")
plt.show()


# -------------------------------------------------------------------
# 2) LOAD TRAINING STATS & PLOT TRAINING TIME
# -------------------------------------------------------------------
with open("train_stats/full_fine_tuning_training_stats.json") as f1:
    json1 = json.load(f1)
with open("train_stats/peft_training_stats.json") as f2:
    json2 = json.load(f2)

methods     = ['Full Fine-tune', 'PEFT (LoRA)']
train_times = [json1['training_time_seconds'], json2['training_time_seconds']]

plt.figure(figsize=(6,4))
bars = plt.bar(methods, train_times, color=['#4C72B0','#55A868'])
plt.ylabel("Training Time (s)")
plt.title("Full vs PEFT Training Time")

for i, v in enumerate(train_times):
    plt.text(i, v + 0.01*max(train_times), f"{v:.1f}s", ha='center')

plt.tight_layout()
plt.savefig("figs/training_time_comparison.png")
plt.show()


# -------------------------------------------------------------------
# 3) PIE CHART: FULL vs PEFT TRAINABLE PARAMETERS
# -------------------------------------------------------------------
full_trainable = json1['trainable_parameters']
peft_trainable = json2['trainable_parameters']

labels = ['Full Fine-tune', 'PEFT (LoRA)']
sizes  = [full_trainable, peft_trainable]
colors = ['#4C72B0', '#55A868']

plt.figure(figsize=(6,6))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(sizes)):,})",
    startangle=90
)
plt.title("Trainable Parameters: Full Fine-tune vs PEFT")
plt.axis('equal')
plt.tight_layout()
plt.savefig("figs/parameters_comparison.png")
plt.show()