"""
Model interpretability for Amharic NER using SHAP and LIME.
Provides: 
- SHAP explanations for token/entity importance
- LIME explanations for local interpretability
- Example difficult case analysis and reporting
"""

# interpretability.py
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
import os

MODEL_PATH = './best-amharic-ner-model'  # Update if needed

# Load model and tokenizer
ner_pipe = pipeline('ner', model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple")

def explain_with_shap(text):
    print(f"\nSHAP explanation for: {text}")
    explainer = shap.Explainer(ner_pipe)
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])

def lime_predict(texts):
    # LIME expects a list of texts and returns probability for each class
    # For NER, we can adapt by returning the probability of entity presence per token
    results = []
    for text in texts:
        entities = ner_pipe(text)
        # Example: return 1 if any entity found, else 0
        results.append([1 if entities else 0, 0 if entities else 1])
    return np.array(results)

def explain_with_lime(text):
    print(f"\nLIME explanation for: {text}")
    class_names = ['Entity', 'No Entity']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, lime_predict, num_features=10)
    exp.show_in_notebook(text=True)
    return exp

def analyze_difficult_cases(inference_file='inference_results.csv', n=5):
    if not os.path.exists(inference_file):
        print(f"File {inference_file} not found. Run inference.py first.")
        return
    df = pd.read_csv(inference_file)
    # Example: Find cases with no entities or too many entities (potential errors)
    df['num_entities'] = df['entities'].apply(lambda x: str(x).count("'entity_group'"))
    print("\n--- Difficult Cases (No entities detected) ---")
    print(df[df['num_entities'] == 0].head(n)[['original_text', 'processed_text']])
    print("\n--- Difficult Cases (Many entities detected) ---")
    print(df[df['num_entities'] > 3].head(n)[['original_text', 'processed_text', 'entities']])
    # Optionally, run SHAP/LIME on these cases

if __name__ == "__main__":
    example = "የልጆች የመጠጥ ቦታ ዋጋ 1000 ብር"
    explain_with_shap(example)
    explain_with_lime(example)
    analyze_difficult_cases()
    print("\nInterpretability analysis complete. Use this script to generate visuals and reports for your final submission.")
