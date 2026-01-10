import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
import requests
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article

# =============================
# DATA LOADING
# =============================
def load_stock(stock):
    stock = stock.upper() + ".NS"
    today = date.today()
    startd = today - timedelta(days=365 * 3)

    df = yf.download(stock, start=startd, end=today)
    df1 = yf.download("^NSEI", period="3y")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df1.columns, pd.MultiIndex):
        df1.columns = df1.columns.get_level_values(0)

    return df, df1

# =============================
# PLOTS
# =============================
def show_return(df):
    df = df.reset_index()
    return px.line(df, x="Date", y="Close", title="Close Price")

def comparison(df, df1):
    df, df1 = df.reset_index(), df1.reset_index()
    df["Norm"] = df["Close"] / df["Close"].iloc[0] * 100
    df1["Norm"] = df1["Close"] / df1["Close"].iloc[0] * 100
    comp = pd.DataFrame({"Date": df["Date"], "Stock": df["Norm"], "NIFTY50": df1["Norm"]})
    return px.line(comp, x="Date", y=["Stock", "NIFTY50"])

def daily_return(df):
    df["Daily Return"] = df["Close"].pct_change()
    return df

def plot_daily_return(df):
    return px.line(df.reset_index(), x="Date", y="Daily Return")

def monthly_return(df):
    return df["Close"].resample("M").last().pct_change().to_frame("monthly_return")

def plot_monthly_return(df):
    return px.line(df.reset_index(), x="Date", y="monthly_return")

def yearly_return(df):
    return df["Close"].resample("Y").last().pct_change().to_frame("yearly_return")

def plot_yearly_return(df):
    return px.line(df.reset_index(), x="Date", y="yearly_return")

# =============================
# INFO
# =============================
def info(monthly_df):
    return {
        "avg_positive": monthly_df[monthly_df > 0].mean().iloc[0] * 100,
        "avg_negative": monthly_df[monthly_df < 0].mean().iloc[0] * 100,
        "win_rate": (monthly_df > 0).mean().iloc[0] * 100
    }

def fundamental_info(stock):
    info = yf.Ticker(stock + ".NS").info
    return info.get("trailingPE"), info.get("sector")

# =============================
# TECHNICALS
# =============================
def ema_macd_signal(df):
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    macd = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    return {
        "trend": "Bullish" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Bearish",
        "latest_macd": macd.iloc[-1]
    }

# =============================
# ML
# =============================
def create_features(df):
    df["return_1"] = df["Close"].pct_change()
    df["ema_ratio"] = df["Close"].ewm(20).mean() / df["Close"].ewm(50).mean()
    df["volatility"] = df["return_1"].rolling(10).std()
    df.dropna(inplace=True)
    return df

def create_targets(df):
    df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["direction"] = (df["future_return"] > 0).astype(int)
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_models(df, features):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBClassifier

    X = df[features]
    y_reg = df["future_return"]
    y_clf = df["direction"]

    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X, y_reg)

    xgb = XGBClassifier(eval_metric="logloss")
    xgb.fit(X, y_clf)

    return rf, xgb, None, None, None

def predict_next_day(df, rf, xgb, features):
    X = df[features].iloc[-1:]
    ret = rf.predict(X)[0]
    direction = xgb.predict(X)[0]
    price = df["Close"].iloc[-1] * (1 + ret)
    return {
        "direction": "UP 📈" if direction else "DOWN 📉",
        "return_pct": ret * 100,
        "predicted_price": price
    }

def predict_n_day_price(price, daily_ret, days):
    return price * ((1 + daily_ret) ** days)

# =============================
# FINBERT
# =============================
@st.cache_resource
def load_finbert():
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tok, mdl

tokenizer, model = load_finbert()
labels = ["negative", "neutral", "positive"]

def analyze_news_sentiment_finbert(urls):
    sentiments = []

    for url in urls:
        try:
            art = Article(url)
            art.download()
            art.parse()
            text = art.title + ". " + art.title + ". " + art.text[:2000]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inputs)

            probs = torch.softmax(out.logits, dim=1)[0].numpy()
            sentiments.append(labels[np.argmax(probs)])
        except:
            pass

    if not sentiments:
        return "Neutral ⚪", 0.0

    return (
        ("Bearish 🔴", sentiments.count("negative") / len(sentiments))
        if sentiments.count("negative") > sentiments.count("positive")
        else ("Bullish 🟢", sentiments.count("positive") / len(sentiments))
    )

# =============================
# NIFTY50
# =============================
def get_nifty50_list():
    data = [
        ("Reliance Industries", "RELIANCE.NS"),
        ("TCS", "TCS.NS"),
        ("HDFC Bank", "HDFCBANK.NS"),
        ("ICICI Bank", "ICICIBANK.NS"),
        ("Infosys", "INFY.NS"),
        ("ITC", "ITC.NS"),
        ("Larsen & Toubro", "LT.NS"),
        ("Axis Bank", "AXISBANK.NS"),
        ("SBI", "SBIN.NS"),
        ("HUL", "HINDUNILVR.NS"),
    ]

    return pd.DataFrame(data, columns=["Company", "Yahoo_Symbol"])

import pandas as pd
import yfinance as yf
import numpy as np

# =============================
# TECHNICAL INDICATORS
# =============================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# =============================
# NIFTY50 REVERSAL SCANNER
# =============================
def scan_nifty_reversals(stocks_df):
    """
    Scan NIFTY50 stocks for bullish reversal
    Conditions:
    - RSI < 35 (oversold)
    - EMA20 > EMA50 OR fresh bullish crossover
    """

    results = []

    for _, row in stocks_df.iterrows():
        symbol = row["Yahoo_Symbol"]
        company = row["Company"]

        try:
            df = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False
            )

            if df.empty or len(df) < 50:
                continue

            df["RSI"] = rsi(df["Close"])
            df["EMA20"] = ema(df["Close"], 20)
            df["EMA50"] = ema(df["Close"], 50)

            if df[["RSI", "EMA20", "EMA50"]].iloc[-2:].isna().any().any():
                continue

            rsi_now = df["RSI"].iloc[-1]
            ema20_now = df["EMA20"].iloc[-1]
            ema50_now = df["EMA50"].iloc[-1]

            ema20_prev = df["EMA20"].iloc[-2]
            ema50_prev = df["EMA50"].iloc[-2]

            # Reversal conditions
            rsi_oversold = rsi_now < 35
            bullish_alignment = ema20_now > ema50_now
            bullish_cross = ema20_prev <= ema50_prev and ema20_now > ema50_now

            if rsi_oversold and bullish_alignment:
                signal = "🔥 Strong Reversal" if bullish_cross else "✅ Reversal"

                results.append({
                    "Company": company,
                    "Symbol": symbol,
                    "RSI": round(rsi_now, 2),
                    "EMA20": round(ema20_now, 2),
                    "EMA50": round(ema50_now, 2),
                    "Signal": signal
                })

        except Exception:
            continue

    return pd.DataFrame(results)




    

