!pip install opendatasets transformers tqdm --quiet

from google.colab import drive
import torch
import os
import tensorflow as tf
import pandas as pd
import shutil
import glob
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import opendatasets as od

drive.mount('/content/drive')

PROJECT_PATH = "/content/drive/My Drive/StockPrediction_2024_FINAL"
NEWS_PATH = os.path.join(PROJECT_PATH, "data_final", "news")
DATA_PATH = os.path.join(PROJECT_PATH, "data_final")

os.makedirs(NEWS_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Computational Device: {device.upper()}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled")
else:
    print("Running on CPU")


dataset_url = 'https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset'
od.download(dataset_url, data_dir=NEWS_PATH)

download_folder = os.path.join(NEWS_PATH, "india-headlines-news-dataset")
csv_files = glob.glob(os.path.join(download_folder, "*.csv"))

if csv_files:
    final_path = os.path.join(NEWS_PATH, "india-news-headlines.csv")
    shutil.move(csv_files[0], final_path)


file_path = os.path.join(NEWS_PATH, "india-news-headlines.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found")

keywords = [
    "Reliance", "TCS", "HDFC", "Infosys", "ICICI", "SBI", "Adani",
    "Nifty", "Sensex", "Bank Nifty", "Gold", "Market",
    "Economy", "RBI", "Inflation", "GDP", "Profit", "Loss"
]
pattern = '|'.join(keywords)

chunks = []
chunk_size = 200000

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    if 'publish_date' in chunk.columns:
        chunk.rename(columns={'publish_date': 'Date', 'headline_text': 'Title'}, inplace=True)

    chunk['Title'] = chunk['Title'].astype(str)
    relevant = chunk[chunk['Title'].str.contains(pattern, case=False, na=False)]

    if not relevant.empty:
        chunks.append(relevant)

if not chunks:
    raise ValueError("No relevant financial news found")

df_relevant = pd.concat(chunks, ignore_index=True)
df_relevant['Date'] = pd.to_datetime(df_relevant['Date'], format='%Y%m%d', errors='coerce')
df_relevant.sort_values('Date', inplace=True)

cleaned_path = os.path.join(NEWS_PATH, "relevant_news_cleaned.csv")
df_relevant.to_csv(cleaned_path, index=False)

print(f"Total filtered headlines: {len(df_relevant)}")


tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

pipe_device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=pipe_device)

print("FinBERT model loaded")


if len(df_relevant) > 30000:
    df_processing = df_relevant.tail(30000).copy()
else:
    df_processing = df_relevant.copy()

headlines = df_processing['Title'].tolist()
dates = df_processing['Date'].tolist()

results = []
batch_size = 128

for i in tqdm(range(0, len(headlines), batch_size)):
    batch_text = headlines[i:i+batch_size]
    batch_dates = dates[i:i+batch_size]

    preds = nlp(batch_text, padding=True, truncation=True, max_length=64)

    for j, pred in enumerate(preds):
        score = pred['score']
        label = pred['label']
        value = score if label == 'positive' else -score if label == 'negative' else 0

        results.append({
            'Date': batch_dates[j],
            'Sentiment_Score': value
        })

df_scores = pd.DataFrame(results)


daily_sentiment = df_scores.groupby(df_scores['Date'].dt.date)['Sentiment_Score'].mean().reset_index()
daily_sentiment.columns = ['Date', 'Daily_Sentiment_Score']
daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

final_path = os.path.join(DATA_PATH, "daily_sentiment_final.csv")
daily_sentiment.to_csv(final_path, index=False)

print("Daily sentiment feature created")

if len(df_scores) > 0:
    enhanced_sentiment = df_scores.groupby(df_scores['Date'].dt.date).agg({
        'Sentiment_Score': ['mean', 'std', 'count']
    }).reset_index()

    enhanced_sentiment.columns = [
        'Date', 'Daily_Sentiment_Score',
        'Sentiment_Volatility', 'News_Volume'
    ]

    enhanced_sentiment['Date'] = pd.to_datetime(enhanced_sentiment['Date'])
    enhanced_sentiment['Sentiment_MA_5'] = enhanced_sentiment['Daily_Sentiment_Score'].rolling(5).mean()
    enhanced_sentiment['Sentiment_Momentum'] = (
        enhanced_sentiment['Daily_Sentiment_Score'] -
        enhanced_sentiment['Sentiment_MA_5']
    )
    enhanced_sentiment['News_Volume_MA'] = enhanced_sentiment['News_Volume'].rolling(5).mean()

    enhanced_sentiment = enhanced_sentiment.ffill().bfill().fillna(0)

    enhanced_path = os.path.join(DATA_PATH, "enhanced_daily_sentiment.csv")
    enhanced_sentiment.to_csv(enhanced_path, index=False)

    print("Enhanced sentiment features created")
else:
    print("No sentiment scores available")
