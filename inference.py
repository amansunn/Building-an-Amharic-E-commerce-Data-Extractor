def read_conll(filename):
    sentences = []
    labels = []
    with open(filename, encoding='utf-8') as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, tag = splits
                    tokens.append(token)
                    tags.append(tag)
        # Add last sentence if file doesn't end with a blank line
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

# Usage
sentences, ner_tags = read_conll('conll_annotation.txt')
print(sentences[0])
print(ner_tags[0])


from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Split into train and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, ner_tags, test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({'tokens': train_sentences, 'ner_tags': train_labels})
val_dataset = Dataset.from_dict({'tokens': val_sentences, 'ner_tags': val_labels})
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

print(dataset)


from transformers import AutoTokenizer

# Use the model checkpoint you want, e.g., XLM-Roberta
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_list = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']
label_to_id = {l: i for i, l in enumerate(label_list)}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] if label[word_idx].startswith('I-') else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
print(tokenized_datasets)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification
metric = evaluate.load("seqeval")

# Load pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list)
)

# Define metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



# Training arguments
from transformers import TrainingArguments
from transformers import AutoModelForTokenClassification

model_checkpoint = "bert-base-cased"

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list)
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

print("✅ TrainingArguments initialized successfully")


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

xlmr_results = train_and_evaluate("xlm-roberta-base")

distilbert_results = train_and_evaluate("distilbert-base-multilingual-cased")

mbert_results = train_and_evaluate("bert-base-multilingual-cased")

import pandas as pd

results_df = pd.DataFrame([
    {"Model": "xlm-roberta-base", **xlmr_results},
    {"Model": "distilbert-base-multilingual-cased", **distilbert_results},
    {"Model": "bert-base-multilingual-cased", **mbert_results},
])
display(results_df)


import shap
from transformers import pipeline

# Load your fine-tuned model and tokenizer
ner_pipe = pipeline("ner", model="./best-amharic-ner-model", tokenizer=tokenizer, aggregation_strategy="simple")

# Example Amharic sentence
example_text = "የልጆች የመጠጥ ቦታ ዋጋ 1000 ብር"

# SHAP explanation
explainer = shap.Explainer(ner_pipe)
shap_values = explainer([example_text])
shap.plots.text(shap_values[0])


import pandas as pd
from datetime import datetime

# Load your data
df = pd.read_csv('conll_annotatio')  # or your latest preprocessed file

# Convert timestamp to datetime
df['date'] = pd.to_datetime(df['date'])

# Group by vendor/channel
vendor_stats = []
for channel, group in df.groupby('channel'):
    # Posting frequency (posts per week)
    weeks = (group['date'].max() - group['date'].min()).days / 7 or 1
    posts_per_week = len(group) / weeks

    # Average views per post
    avg_views = group['views'].mean() if 'views' in group else None

    # Top performing post
    top_post = group.loc[group['views'].idxmax()] if 'views' in group else group.iloc[0]
    top_product = top_post['text']
    # Use your NER model to extract product and price from top_product

    # Average price point (use NER model to extract all prices)
    # Example: Assume you have a function extract_prices(text) that returns a list of prices
    all_prices = []
    for msg in group['text']:
        # prices = extract_prices(msg)  # Use your NER model here
        # all_prices.extend(prices)
        pass  # Replace with actual extraction
    avg_price = sum(all_prices) / len(all_prices) if all_prices else None

    # Lending Score (example formula)
    lending_score = (avg_views or 0) * 0.5 + posts_per_week * 0.5

    vendor_stats.append({
        'Vendor': channel,
        'Avg. Views/Post': avg_views,
        'Posts/Week': posts_per_week,
        'Avg. Price (ETB)': avg_price,
        'Lending Score': lending_score,
        'Top Post': top_product
    })

scorecard = pd.DataFrame(vendor_stats)
print(scorecard[['Vendor', 'Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)', 'Lending Score']])

"""
Run NER inference on new Amharic messages using the best fine-tuned model.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = './best-amharic-ner-model'  # Update if needed

# Load model and tokenizer
ner_pipe = pipeline('ner', model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple")

def extract_entities(text):
    return ner_pipe(text)

if __name__ == "__main__":
    example = "የልጆች የመጠጥ ቦታ ዋጋ 1000 ብር"
    print(f"Entities in: {example}")
    print(extract_entities(example))