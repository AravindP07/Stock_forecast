# Macro-Sentiment Stock Forecasting using Bi-GRU
Macro-Sentiment Stock Forecasting using Bi-GRU Ensemble

A state-of-the-art Financial Time Series Forecasting System designed for Indian equity markets. This project develops a macro-aware deep learning framework using a Bi-Directional GRU ensemble architecture combined with advanced feature engineering to predict next-day stock prices with high stability and adaptability across multiple sectors.

Key Features :

Macro-Aware Modeling:
Integrates USD/INR exchange rate and crude oil prices to capture external macroeconomic influences on Indian equities.

Sector-Specific Feature Engineering:
Applies customized technical indicators and volatility features tailored for banking, auto, pharma, and diversified sectors.

Dynamic Lookback Mechanism:
Automatically adjusts historical window size based on asset volatility to improve short-term predictive stability.

Regularized Deep Learning Architecture:
Implements Bi-Directional GRU with L2 regularization and dropout to prevent overfitting and enhance generalization.

Interactive Forecasting Dashboard:
Professional Streamlit-based UI enabling real-time next-day price prediction, performance metrics visualization, and market factor insights.

Tech Stack:

Core AI:
Python, TensorFlow/Keras (Bi-GRU Architecture)

Financial Data Processing:
Pandas, NumPy, yFinance, TA (Technical Analysis Library)

Model Evaluation:
Scikit-learn (MAPE, RMSE, MAE, R²)

Visualization:
Matplotlib, Plotly

Deployment:
Streamlit (Web Application)

Dataset :

The model was trained on historical Indian equity market data and macroeconomic indicators.

Stock Data Source: Yahoo Finance
(https://finance.yahoo.com
)

Macro Data:
USD/INR Exchange Rate (INR=X)
Brent Crude Oil Futures (BZ=F)

Time Range: 2015 – 2025

Assets Covered:
50 Nifty stocks + 10 ETFs (Total 57 assets)

Features Engineered:
Technical indicators (RSI, SMA, MACD, Volatility), Log Returns, Momentum features, Macro indicators.

Model Architecture :

We designed a hybrid deep learning forecasting architecture optimized for financial time-series prediction:

Base Model:
Bi-Directional GRU (64 units) to capture forward and backward temporal dependencies.

Sequential Layering:
Dropout layer for regularization followed by a Uni-Directional GRU (32 units) to refine learned temporal patterns.

Output Layer:
Dense(1) regression node predicting next-day log return.

Feature Fusion:
Price, momentum, volatility, and macroeconomic indicators are concatenated before sequence modeling.

Normalization:
RobustScaler applied to reduce sensitivity to extreme market shocks (e.g., COVID crash).

Performance Highlights:

The system demonstrates strong predictive consistency across diverse market regimes:

High R² scores across low and high volatility stocks

Stable validation loss convergence

Low MAPE and RMSE across 57 assets

Effective adaptation during extreme market events (e.g., COVID period)

Observation:
The model effectively captures short-term trend continuation and volatility clustering while maintaining generalization across multiple sectors.

