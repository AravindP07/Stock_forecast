!pip install yfinance ta --quiet

from google.colab import drive
import os
import yfinance as yf
import pandas as pd
import numpy as np
import glob
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import matplotlib.pyplot as plt
import random

drive.mount('/content/drive')

PROJECT_PATH = "/content/drive/My Drive/StockPrediction_2024_FINAL"
DATA_PATH = os.path.join(PROJECT_PATH, "data_final")
STOCK_PATH = os.path.join(DATA_PATH, "stocks")
NEWS_PATH = os.path.join(DATA_PATH, "news")

os.makedirs(STOCK_PATH, exist_ok=True)
os.makedirs(NEWS_PATH, exist_ok=True)

print("Project environment initialized")

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

nifty_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LTIM.NS", "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "WIPRO.NS", "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "GRASIM.NS", "CIPLA.NS", "DIVISLAB.NS", "COALINDIA.NS", "BPCL.NS",
    "TATAMOTORS.NS", "HINDALCO.NS", "EICHERMOT.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "TECHM.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "SBILIFE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "INDUSINDBK.NS", "TATACONSUM.NS", "UPL.NS"
]

etf_tickers = [
    "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS",
    "LIQUIDBEES.NS", "CPSEETF.NS", "MON100.NS", "HDFCNIFTY.NS",
    "AXISNIFTY.NS", "ITBEES.NS"
]

ALL_TICKERS = nifty_50_tickers + etf_tickers
print(f"Total assets: {len(ALL_TICKERS)}")

success_count = 0
failed_tickers = []

for ticker in ALL_TICKERS:
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True
        )

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(os.path.join(STOCK_PATH, f"{ticker}.csv"))
            success_count += 1
        else:
            failed_tickers.append(ticker)

        time.sleep(0.2)

    except Exception:
        failed_tickers.append(ticker)

print(f"Download completed. Success: {success_count}")

def add_sector_specific_features(df, ticker):

    if 'Log_Returns' in df.columns:
        df['Volatility_5D'] = df['Log_Returns'].rolling(5).std()
        df['Volatility_20D'] = df['Log_Returns'].rolling(20).std()
        df['Volatility_Ratio'] = df['Volatility_5D'] / df['Volatility_20D']

    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Price_Acceleration'] = df['Momentum_5'] - df['Momentum_10']
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    if not any(x in ticker for x in ['ETF', 'BEES']):
        df['SMA_20'] = df['Close'].rolling(20).mean()

    if any(x in ticker for x in ['BANK', 'FIN', 'HDFC', 'ICICI', 'AXIS']):
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['Yield_Curve_Proxy'] = (df['Close'] / df['SMA_50']) / (df['Close'] / df['SMA_200'])

    if 'SMA_50' in df.columns:
        df['Above_SMA_50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['SMA_Distance'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    return df


def process_stock_data(file_path):

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]

    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    df = add_sector_specific_features(df, file_path)

    essential_cols = ['Close', 'RSI', 'SMA_50', 'Log_Returns']
    df = df.dropna(subset=[col for col in essential_cols if col in df.columns])

    return df

stock_files = glob.glob(os.path.join(STOCK_PATH, "*.csv"))

for file in stock_files:
    try:
        df_clean = process_stock_data(file)
        df_clean.to_csv(file)
    except Exception:
        continue

print("Feature engineering completed")

def validate_dataset_integrity():

    stock_files = glob.glob(os.path.join(STOCK_PATH, "*.csv"))
    if not stock_files:
        print("No stock files found")
        return

    test_file = random.choice(stock_files)
    df = pd.read_csv(test_file, index_col=0, parse_dates=True)

    if len(df) > 100:
        sample = df.tail(100)

        plt.figure(figsize=(12, 6))
        plt.plot(sample.index, sample['Close'], label='Close')
        if 'SMA_50' in df.columns:
            plt.plot(sample.index, sample['SMA_50'], linestyle='--', label='SMA 50')

        plt.title("Sample Validation")
        plt.legend()
        plt.tight_layout()
        plt.show()

validate_dataset_integrity()
