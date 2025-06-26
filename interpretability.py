# interpretability.py
"""
Model interpretability for Amharic NER using SHAP and LIME.
"""
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = './best-amharic-ner-model'  # Update if needed

# Load model and tokenizer
ner_pipe = pipeline('ner', model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple")

def explain_with_shap(text):
    explainer = shap.Explainer(ner_pipe)
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])

# LIME for NER is more complex; here is a basic example for text classification
from lime.lime_text import LimeTextExplainer
import numpy as np

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
    class_names = ['Entity', 'No Entity']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, lime_predict, num_features=10)
    exp.show_in_notebook(text=True)

if __name__ == "__main__":
    example = "የልጆች የመጠጥ ቦታ ዋጋ 1000 ብር"
    print("SHAP explanation:")
    explain_with_shap(example)
    print("LIME explanation:")
    explain_with_lime(example)
