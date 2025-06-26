# README.md

# Amharic E-commerce Data Extractor: End-to-End Pipeline

This project provides a complete pipeline for extracting, labeling, fine-tuning, and analyzing Amharic e-commerce data from Telegram channels. It includes data ingestion, preprocessing, NER labeling, model training, interpretability, and vendor analytics for micro-lending.

## Project Structure

- `data_ingestion.py` — Scrape and collect Telegram messages (text, images, metadata)
- `data_cleaning.py` — Preprocess and normalize Amharic text
- `label_ner_data.py` — Label data for NER in CoNLL format
- `fine_tune_ner.py` — Fine-tune and evaluate NER models (XLM-Roberta, DistilBERT, mBERT)
- `inference.py` — Run NER inference on new messages
- `vendor_analysis.py` — Analyze vendors and generate a lending scorecard
- `requirements.txt` — All required Python packages
- `processed_amharic_data.csv`, `conll_annotation.txt` — Example data files

## How to Run the Pipeline

1. **Data Ingestion:**
   - Run `data_ingestion.py` to collect Telegram data into `collected_messages.csv`.
2. **Preprocessing:**
   - Run `data_cleaning.py` to clean and normalize text, outputting `processed_amharic_data.csv`.
3. **NER Labeling:**
   - Use `label_ner_data.py` to annotate a subset of data in CoNLL format (`conll_annotation.txt`).
4. **Model Fine-Tuning:**
   - Run `fine_tune_ner.py` to train and evaluate NER models. Compare results and select the best model.
5. **Inference:**
   - Use `inference.py` to extract entities from new messages using the best model.
6. **Vendor Analytics:**
   - Run `vendor_analysis.py` to generate a vendor scorecard for micro-lending decisions.

## Requirements

See `requirements.txt` for all dependencies. Install with:
```
pip install -r requirements.txt
```

## Documentation
- Each script is documented with comments and usage instructions.
- The pipeline is modular and reproducible.

---

For questions or improvements, please see the code comments or contact the project maintainer.
