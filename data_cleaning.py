import pandas as pd
import re
from typing import List

# --- Function Definitions ---

def normalize_amharic(text: str) -> str:
    """
    Normalizes variations of Amharic characters to a standard form.
    """
    normalization_map = {
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        'ጸ': 'ፀ', 'ጹ': 'ፁ', 'ጺ': 'ፂ', 'ጻ': 'ፃ', 'ጼ': 'ፄ', 'ጽ': 'ፅ', 'ጾ': 'ፆ',
    }
    regex = re.compile("|".join(map(re.escape, normalization_map.keys())))
    text = regex.sub(lambda match: normalization_map[match.group(0)], text)
    # Labialized normalization (e.g., ሉዋ -> ሏ)
    labialized_patterns = [
        (r'(ሉ[ዋአ])', 'ሏ'), (r'(ሙ[ዋአ])', 'ሟ'), (r'(ቱ[ዋአ])', 'ቷ'), (r'(ሩ[ዋአ])', 'ሯ'),
        (r'(ሱ[ዋአ])', 'ሷ'), (r'(ሹ[ዋአ])', 'ሿ'), (r'(ቁ[ዋአ])', 'ቋ'), (r'(ቡ[ዋአ])', 'ቧ'),
        (r'(ቹ[ዋአ])', 'ቿ'), (r'(ሁ[ዋአ])', 'ኋ'), (r'(ኑ[ዋአ])', 'ኗ'), (r'(ኙ[ዋአ])', 'ኟ'),
        (r'(ኩ[ዋአ])', 'ኳ'), (r'(ዙ[ዋአ])', 'ዟ'), (r'(ጉ[ዋአ])', 'ጓ'), (r'(ደ[ዋአ])', 'ዷ'),
        (r'(ጡ[ዋአ])', 'ጧ'), (r'(ጩ[ዋአ])', 'ጯ'), (r'(ጹ[ዋአ])', 'ጿ'), (r'(ፉ[ዋአ])', 'ፏ'),
    ]
    for pat, rep in labialized_patterns:
        text = re.sub(pat, rep, text)
    text = re.sub('[ቊ]', 'ቁ', text)
    text = re.sub('[ኵ]', 'ኩ', text)
    return text

def clean_text(text: str) -> str:
    """
    Cleans the text by removing URLs, mentions, hashtags, and unwanted characters,
    while keeping both Amharic and English text.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r't\.me\/\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\u1200-\u137fa-z\d\s።፡,.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes Amharic and English text using whitespace and basic punctuation.
    """
    # Split on whitespace and punctuation (except Amharic/English letters and numbers)
    tokens = re.findall(r'[\u1200-\u137fa-zA-Z\d]+', text)
    return tokens

def is_relevant_message(text: str) -> bool:
    """
    Filters out irrelevant or low-quality messages (e.g., too short, empty, or spam).
    """
    if not text or len(text.strip()) < 5:
        return False
    # Add more rules as needed (e.g., filter out certain keywords)
    return True

# --- Main Processing ---

def main():
    """
    Main data cleaning pipeline: loads data, normalizes, cleans, tokenizes, filters, and saves output.
    """
    try:
        df = pd.read_csv('collected_messages.csv')
        print("Successfully loaded data.")
    except FileNotFoundError:
        print("Error: Make sure your CSV file is in the same directory or provide the full path.")
        exit()

    # If 'Message Text' column does not exist, try 'text' (from telegram_scraper.py)
    if 'Message Text' in df.columns:
        df['processed_text'] = df['Message Text'].fillna('').astype(str)
    elif 'text' in df.columns:
        df['processed_text'] = df['text'].fillna('').astype(str)
    else:
        print("Error: No suitable text column found in the CSV file.")
        exit()

    print("Normalizing text...")
    df['processed_text'] = df['processed_text'].apply(normalize_amharic)
    print("Cleaning text (now keeping English)...")
    df['processed_text'] = df['processed_text'].apply(clean_text)
    print("Filtering irrelevant messages...")
    df = df[df['processed_text'].apply(is_relevant_message)].copy()
    print("Tokenizing text...")
    df['tokens'] = df['processed_text'].apply(tokenize_text)
    output_filename = 'processed_amharic_data.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\nPreprocessing complete! The cleaned data has been saved to '{output_filename}'")
    print("\nHere is a sample of your original vs. processed text and tokens:")
    if 'Message Text' in df.columns:
        print(df[['Message Text', 'processed_text', 'tokens']].head(10).to_string())
    elif 'text' in df.columns:
        print(df[['text', 'processed_text', 'tokens']].head(10).to_string())

if __name__ == "__main__":
    main()