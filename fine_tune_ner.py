# fine_tune_ner.py
"""
Fine-tune and evaluate NER models (XLM-Roberta, DistilBERT, mBERT) on Amharic data.
"""
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# Load and parse CoNLL data
def read_conll(filename):
    sentences, labels = [], []
    with open(filename, encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, tag = splits
                    tokens.append(token)
                    tags.append(tag)
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

sentences, ner_tags = read_conll('conll_annotation.txt')
train_s, val_s, train_l, val_l = train_test_split(sentences, ner_tags, test_size=0.2, random_state=42)
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

def train_and_evaluate(model_checkpoint):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = Dataset.from_dict({'tokens': train_s, 'ner_tags': train_l})
    val_dataset = Dataset.from_dict({'tokens': val_s, 'ner_tags': val_l})
    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
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
    training_args = TrainingArguments(
        output_dir=f"./results-{model_checkpoint.replace('/', '-')}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    print(f"Results for {model_checkpoint}:")
    print(results)
    return results

if __name__ == "__main__":
    results = {}
    for model_checkpoint in [
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased"
    ]:
        results[model_checkpoint] = train_and_evaluate(model_checkpoint)
    pd.DataFrame([
        {"Model": k, **v} for k, v in results.items()
    ]).to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to model_comparison.csv")
