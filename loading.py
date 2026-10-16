
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import torch

from datetime import date, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
import plotly.express as px


# =============================
# HELPERS
# =============================

def flatten_columns(df):
    """Handle MultiIndex columns returned by newer yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if "Close" in values:
                df.columns = values
                break
        else:
            df.columns = df.columns.get_level_values(0)

    return df


def normalize_symbol(stock):
    """Normalize an NSE symbol without adding .NS twice."""
    stock = stock.strip().upper()

    if not stock:
        raise ValueError("Please enter a valid NSE stock symbol.")

    if not stock.endswith(".NS"):
        stock += ".NS"

    return stock


# =============================
# DATA LOADING
# =============================

def load_stock(stock):
    stock = normalize_symbol(stock)

    today = date.today()
    startd = today - timedelta(days=365 * 3)

    df = yf.download(
        stock,
        start=startd,
        end=today,
        progress=False,
        auto_adjust=True
    )

    df1 = yf.download(
        "^NSEI",
        period="3y",
        progress=False,
        auto_adjust=True
    )

    df = flatten_columns(df)
    df1 = flatten_columns(df1)

    if df.empty:
        raise ValueError(
            f"No stock data found for {stock}. Check the symbol."
        )

    if df1.empty:
        raise ValueError("Unable to download NIFTY 50 data.")

    df.index = pd.to_datetime(df.index)
    df1.index = pd.to_datetime(df1.index)

    df = df.sort_index()
    df1 = df1.sort_index()

    return df, df1


# =============================
# PLOTS
# =============================

def show_return(df):
    data = df.reset_index()
    return px.line(
        data,
        x="Date",
        y="Close",
        title="Close Price"
    )


def comparison(df, df1):
    stock_data = df[["Close"]].copy()
    nifty_data = df1[["Close"]].copy()

    stock_data["Stock"] = (
        stock_data["Close"] / stock_data["Close"].iloc[0] * 100
    )

    nifty_data["NIFTY50"] = (
        nifty_data["Close"] / nifty_data["Close"].iloc[0] * 100
    )

    stock_data = stock_data[["Stock"]]
    nifty_data = nifty_data[["NIFTY50"]]

    # Align both datasets using common trading dates
    comp = stock_data.join(nifty_data, how="inner").reset_index()

    return px.line(
        comp,
        x="Date",
        y=["Stock", "NIFTY50"],
        title="Stock vs NIFTY 50 (Normalized)"
    )


def daily_return(df):
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change()
    return df


def plot_daily_return(df):
    return px.line(
        df.reset_index(),
        x="Date",
        y="Daily Return",
        title="Daily Returns"
    )


def monthly_return(df):
    return (
        df["Close"]
        .resample("ME")
        .last()
        .pct_change()
        .to_frame("monthly_return")
    )


def plot_monthly_return(df):
    return px.line(
        df.reset_index(),
        x="Date",
        y="monthly_return",
        title="Monthly Returns"
    )


def yearly_return(df):
    return (
        df["Close"]
        .resample("YE")
        .last()
        .pct_change()
        .to_frame("yearly_return")
    )


def plot_yearly_return(df):
    return px.line(
        df.reset_index(),
        x="Date",
        y="yearly_return",
        title="Yearly Returns"
    )


# =============================
# RETURN STATISTICS
# =============================

def info(monthly_df):
    returns = monthly_df["monthly_return"].dropna()

    positive = returns[returns > 0]
    negative = returns[returns < 0]

    return {
        "avg_positive": (
            positive.mean() * 100 if not positive.empty else 0.0
        ),
        "avg_negative": (
            negative.mean() * 100 if not negative.empty else 0.0
        ),
        "win_rate": (
            (returns > 0).mean() * 100 if not returns.empty else 0.0
        )
    }


def fundamental_info(stock):
    symbol = normalize_symbol(stock)

    try:
        stock_info = yf.Ticker(symbol).info
        return (
            stock_info.get("trailingPE"),
            stock_info.get("sector")
        )
    except Exception:
        return None, None


# =============================
# TECHNICAL INDICATORS
# =============================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # Handle periods with gains but no losses
    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100.0
    )

    # Handle periods with neither gains nor losses
    result = result.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50.0
    )

    return result


def ema_macd_signal(df):
    if df.empty or len(df) < 2:
        raise ValueError("Insufficient stock data for technical analysis.")

    close = df["Close"]

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    macd = ema(close, 12) - ema(close, 26)

    return {
        "trend": (
            "Bullish"
            if ema50.iloc[-1] > ema200.iloc[-1]
            else "Bearish"
        ),
        "latest_macd": float(macd.iloc[-1])
    }


# =============================
# MACHINE LEARNING
# =============================

def create_features(df):
    df = df.copy()

    close = df["Close"]

    df["return_1"] = close.pct_change()

    df["ema_ratio"] = (
        ema(close, 20) / ema(close, 50)
    )

    df["volatility"] = df["return_1"].rolling(10).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def create_targets(df):
    df = df.copy()

    df["future_return"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    df["direction"] = (
        df["future_return"] > 0
    ).astype(int)

    # Removes the last row because its future return is unknown
    df = df.dropna(subset=["future_return"])

    return df


def train_models(df, features):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBClassifier

    required = features + ["future_return", "direction"]

    data = df.dropna(subset=required).copy()

    if len(data) < 50:
        raise ValueError(
            "Not enough clean historical data to train the models."
        )

    X = data[features]
    y_reg = data["future_return"]
    y_clf = data["direction"]

    if y_clf.nunique() < 2:
        raise ValueError(
            "Training data contains only one direction class."
        )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y_reg)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X, y_clf)

    # Preserve the five-value return expected by app.py
    return rf, xgb, None, None, None


def predict_next_day(df, rf, xgb, features):
    if df.empty:
        raise ValueError("No data available for prediction.")

    X = df[features].iloc[-1:]

    ret = float(rf.predict(X)[0])
    direction = int(xgb.predict(X)[0])

    latest_close = float(df["Close"].iloc[-1])
    price = latest_close * (1 + ret)

    return {
        "direction": "UP 📈" if direction else "DOWN 📉",
        "return_pct": ret * 100,
        "predicted_price": price
    }


def predict_n_day_price(price, daily_ret, days):
    return price * ((1 + daily_ret) ** days)


# =============================
# FINBERT NEWS SENTIMENT
# =============================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_finbert()


def analyze_news_sentiment_finbert(urls):
    all_probabilities = []

    # Use the model's actual label mapping instead of assuming
    # the order of negative, neutral and positive.
    id2label = model.config.id2label

    label_mapping = {
        str(label).lower(): idx
        for idx, label in id2label.items()
    }

    def find_label(candidates):
        for candidate in candidates:
            if candidate in label_mapping:
                return label_mapping[candidate]
        return None

    negative_idx = find_label(["negative"])
    neutral_idx = find_label(["neutral"])
    positive_idx = find_label(["positive"])

    if any(
        idx is None
        for idx in [negative_idx, neutral_idx, positive_idx]
    ):
        return "Neutral ⚪", 0.0

    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()

            text = (
                article.title + ". " + article.text[:2000]
            ).strip()

            if not text:
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                output = model(**inputs)

            probabilities = torch.softmax(
                output.logits,
                dim=1
            )[0].cpu().numpy()

            all_probabilities.append(probabilities)

        except Exception:
            # Skip articles that cannot be downloaded or analyzed
            continue

    if not all_probabilities:
        return "Neutral ⚪", 0.0

    # Average probabilities across successfully analyzed articles
    avg_probs = np.mean(all_probabilities, axis=0)

    sentiment_indices = {
        "negative": negative_idx,
        "neutral": neutral_idx,
        "positive": positive_idx
    }

    winning_sentiment = max(
        sentiment_indices,
        key=lambda label: avg_probs[sentiment_indices[label]]
    )

    confidence = float(
        avg_probs[sentiment_indices[winning_sentiment]]
    )

    sentiment_names = {
        "negative": "Bearish 🔴",
        "neutral": "Neutral ⚪",
        "positive": "Bullish 🟢"
    }

    return sentiment_names[winning_sentiment], confidence


# =============================
# NIFTY50 STOCK LIST
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

    return pd.DataFrame(
        data,
        columns=["Company", "Yahoo_Symbol"]
    )


# =============================
# NIFTY50 REVERSAL SCANNER
# =============================

def scan_nifty_reversals(stocks_df):
    """
    Scan the supplied stock list for potential bullish reversals.

    Conditions:
    - RSI below 35
    - EMA20 above EMA50

    A fresh bullish crossover is labeled Strong Reversal.
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
                progress=False,
                auto_adjust=True
            )

            df = flatten_columns(df)

            if df.empty or len(df) < 50:
                continue

            close = df["Close"]

            df["RSI"] = rsi(close)
            df["EMA20"] = ema(close, 20)
            df["EMA50"] = ema(close, 50)

            latest = df.iloc[-1]
            previous = df.iloc[-2]

            if df[
                ["RSI", "EMA20", "EMA50"]
            ].iloc[-2:].isna().any().any():
                continue

            rsi_now = float(latest["RSI"])
            ema20_now = float(latest["EMA20"])
            ema50_now = float(latest["EMA50"])

            ema20_prev = float(previous["EMA20"])
            ema50_prev = float(previous["EMA50"])

            rsi_oversold = rsi_now < 35
            bullish_alignment = ema20_now > ema50_now

            bullish_cross = (
                ema20_prev <= ema50_prev
                and ema20_now > ema50_now
            )

            if rsi_oversold and bullish_alignment:
                signal = (
                    "🔥 Strong Reversal"
                    if bullish_cross
                    else "✅ Reversal"
                )

                results.append({
                    "Company": company,
                    "Symbol": symbol,
                    "RSI": round(rsi_now, 2),
                    "EMA20": round(ema20_now, 2),
                    "EMA50": round(ema50_now, 2),
                    "Signal": signal
                })

        except Exception:
            # Continue scanning other stocks if one download fails
            continue

    return pd.DataFrame(
        results,
        columns=[
            "Company",
            "Symbol",
            "RSI",
            "EMA20",
            "EMA50",
            "Signal"
        ]
    )





    

