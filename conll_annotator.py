import pandas as pd
import ast

INPUT_FILE = 'processed_amharic_data.csv'
OUTPUT_FILE = 'conll_annotation.txt'
NUM_MESSAGES = 50  # You can change this to 30 if you want fewer

def load_and_prepare():
    df = pd.read_csv(INPUT_FILE)
    # Use 'processed_text' and 'tokens' columns if available
    if 'processed_text' in df.columns and 'tokens' in df.columns:
        messages = df[['processed_text', 'tokens']].head(NUM_MESSAGES)
    elif 'Message Text' in df.columns:
        messages = df[['Message Text']].rename(columns={'Message Text': 'processed_text'}).head(NUM_MESSAGES)
        messages['tokens'] = messages['processed_text'].apply(lambda x: x.split())
    else:
        raise Exception('No suitable text column found.')
    # Convert string representation of list to list if needed
    messages['tokens'] = messages['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    return messages

def main():
    messages = load_and_prepare()
    print(f"Loaded {len(messages)} messages for annotation.")
    print("Next: We will annotate each token in each message.")
    print("Entity labels: B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O")
    print("For each token, type the label (or press Enter for 'O'). Type 'exit' to stop early.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, row in messages.iterrows():
            print(f"\nMessage {idx+1}: {row['processed_text']}")
            for token in row['tokens']:
                while True:
                    label = input(f"Token: {token}  Label: ").strip()
                    if label == '':
                        label = 'O'
                    if label == 'exit':
                        print(f"\nAnnotation stopped. Partial data saved to {OUTPUT_FILE}")
                        return
                    if label in ['B-Product','I-Product','B-LOC','I-LOC','B-PRICE','I-PRICE','O']:
                        f.write(f"{token} {label}\n")
                        break
                    else:
                        print("Invalid label. Choose from: B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O")
            f.write("\n")  # Blank line between messages
    print(f"\nAnnotation complete! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
