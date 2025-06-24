# telegram_scraper.py
"""
Script to connect to Telegram and fetch messages from specified channels.
"""
from telethon import TelegramClient
from config import API_ID, API_HASH, CHANNELS
import asyncio
import pandas as pd
from datetime import datetime

# Output file for collected messages
data_file = 'collected_messages.csv'

async def fetch_messages():
    client = TelegramClient('session_name', API_ID, API_HASH)
    await client.start()
    all_messages = []
    for channel in CHANNELS:
        try:
            async for message in client.iter_messages(channel, limit=100):
                if message.text:
                    all_messages.append({
                        'channel': channel,
                        'sender_id': message.sender_id,
                        'date': message.date,
                        'text': message.text.replace('\n', ' '),
                        'message_id': message.id
                    })
        except Exception as e:
            print(f"Error fetching from {channel}: {e}")
    df = pd.DataFrame(all_messages)
    df.to_csv(data_file, index=False)
    print(f"Saved {len(all_messages)} messages to {data_file}")
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(fetch_messages())
