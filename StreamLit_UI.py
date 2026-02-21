import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="V1 Algo-Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_metrics():
    if os.path.exists("v1_final_metrics.csv"):
        return pd.read_csv("v1_final_metrics.csv")
    return pd.DataFrame()

def load_model_artifacts(ticker, model_folder):
    model_path = os.path.join(model_folder, f"model_{ticker}.keras")
    scaler_path = os.path.join(model_folder, f"scaler_{ticker}.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_stock_data(ticker, stock_folder):
    path = os.path.join(stock_folder, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20D'] = df['Log_Returns'].rolling(20).std()
    df = df.dropna()
    return df

def main():

    st.sidebar.title("AI Forecast Control")

    model_folder = st.sidebar.text_input("Model Directory", "saved_models_v1_final")
    stock_folder = st.sidebar.text_input("Stock Data Directory", "data_final/stocks")

    files = glob.glob(os.path.join(stock_folder, "*.csv"))
    tickers = sorted([os.path.basename(f).replace(".csv", "") for f in files])
    selected_ticker = st.sidebar.selectbox("Select Asset", tickers)

    st.title(f"{selected_ticker} Next-Day Forecast")

    df_metrics = load_metrics()
    if not df_metrics.empty:
        row = df_metrics[df_metrics['Ticker'] == selected_ticker]
        if not row.empty:
            st.metric("Model R² Score", f"{row.iloc[0]['R2_Score']:.2f}")

    model, scaler = load_model_artifacts(selected_ticker, model_folder)
    df = get_stock_data(selected_ticker, stock_folder)

    if model is None or df is None:
        st.warning("Model or Data missing.")
        return

    features = ['Close', 'Log_Returns', 'RSI', 'SMA_50',
                'USD_INR', 'Volatility_20D',
                'Daily_Sentiment_Score', 'News_Volume']

    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    vol = df['Log_Returns'].std()
    lookback = 15
    if vol > 0.025:
        lookback = 10
    if 'ETF' in selected_ticker or 'BEES' in selected_ticker:
        lookback = 20

    last_seq = df[features].iloc[-lookback:]
    input_scaled = scaler.transform(last_seq)
    input_reshaped = input_scaled.reshape(1, lookback, len(features))

    pred_log_ret = model.predict(input_reshaped, verbose=0)[0][0]

    last_close = df['Close'].iloc[-1]
    pred_price = last_close * np.exp(pred_log_ret)
    change_pct = (pred_price - last_close) / last_close * 100

    st.markdown(f"""
    <div style='padding:20px;background:#111;border-radius:10px;text-align:center'>
        <h3>Predicted Close (Next Trading Day)</h3>
        <h1>{pred_price:.2f}</h1>
        <p>Predicted Change: {change_pct:+.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Price Visualization")

    history_window = 35
    recent_df = df.iloc[-history_window:]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=recent_df.index,
        open=recent_df['Open'],
        high=recent_df['High'],
        low=recent_df['Low'],
        close=recent_df['Close'],
        name="Historical Price"
    ))

    next_date = recent_df.index[-1] + pd.Timedelta(days=1)

    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[pred_price],
        mode='markers+text',
        marker=dict(size=14, symbol='star', color='orange'),
        text=[f"{pred_price:.2f}"],
        textposition="top center",
        name="AI Forecast"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_yaxes(autorange=True)

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
