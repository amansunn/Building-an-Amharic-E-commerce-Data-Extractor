# vendor_analysis.py
"""
Analyze vendors and generate a scorecard for micro-lending decisions.
"""
import pandas as pd
from datetime import datetime

# Load preprocessed data with metadata
# Assumes 'collected_messages.csv' has columns: channel, date, views, text

def extract_prices(text):
    # Dummy price extraction: replace with your NER model inference
    import re
    prices = re.findall(r'\d{2,}', text)
    return [int(p) for p in prices]

def analyze_vendor(df, vendor_col='channel'):
    results = []
    for vendor, group in df.groupby(vendor_col):
        weeks = (group['date'].max() - group['date'].min()).days / 7 or 1
        posts_per_week = len(group) / weeks
        avg_views = group['views'].mean() if 'views' in group else None
        # Top performing post
        top_post = group.loc[group['views'].idxmax()] if 'views' in group else group.iloc[0]
        # Average price point (use NER model to extract all prices)
        all_prices = []
        for msg in group['text']:
            all_prices.extend(extract_prices(msg))
        avg_price = sum(all_prices) / len(all_prices) if all_prices else None
        lending_score = (avg_views or 0) * 0.5 + posts_per_week * 0.5
        results.append({
            'Vendor': vendor,
            'Posts/Week': posts_per_week,
            'Avg. Views/Post': avg_views,
            'Avg. Price (ETB)': avg_price,
            'Lending Score': lending_score,
            'Top Post': top_post['text']
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_csv('collected_messages.csv')
    df['date'] = pd.to_datetime(df['date'])
    scorecard = analyze_vendor(df)
    print(scorecard[['Vendor', 'Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)', 'Lending Score']])
    scorecard.to_csv('vendor_scorecard.csv', index=False)
    print("Vendor scorecard saved to vendor_scorecard.csv")
