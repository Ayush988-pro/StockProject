# =============================
# IMPORTS
# =============================
import streamlit as st

from loading import (
    load_stock, show_return, comparison,
    daily_return, plot_daily_return,
    monthly_return, plot_monthly_return,
    yearly_return, plot_yearly_return,
    info, fundamental_info,
    ema_macd_signal,
    create_features, create_targets,
    train_models, predict_next_day,
    predict_n_day_price,
    analyze_news_sentiment_finbert,get_nifty50_list, scan_nifty_reversals
)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    layout="wide"
)

# =============================
# GLOBAL STYLES
# =============================
st.markdown("""
<style>
.stApp { background-color:#0e1117; color:#e0e0e0; }
section[data-testid="stSidebar"] { background-color:#111827; }
h1,h2,h3,h4 { color:#4da3ff; }
div[data-testid="metric-container"] {
    background-color:#111827;
    border:1px solid #1f2937;
    border-radius:8px;
    padding:10px;
}
div.stButton > button { margin:0 auto; display:block; }
</style>
""", unsafe_allow_html=True)

# =============================
# MARKET SENTIMENT (FINBERT)
# =============================
news_urls = [
    "https://www.reuters.com/world/india/indian-shares-fall-as-it-stocks-drag-markets-2024-01-07/","https://www.moneycontrol.com/news/business/markets/sensex-falls-over-400-points-nifty-below-26300-it-stocks-drag-106640997.html"

]



sentiment, score = analyze_news_sentiment_finbert(news_urls)

color_map = {
    "Bullish 🟢": "#00c853",
    "Bearish 🔴": "#ff5252",
    "Neutral ⚪": "#ffca28"
}

color = color_map.get(sentiment, "#ffffff")

st.markdown(
f"""
<div style="background:#0e1117;padding:10px;border-bottom:1px solid #333;color:{color};font-weight:600;">
<marquee scrollamount="6">
📢 Market Sentiment → <b>{sentiment}</b> | Confidence: {score:.2f}
</marquee>
</div>
""",
unsafe_allow_html=True
)

# =============================
# TITLE
# =============================
st.markdown("<h1 style='text-align:center;'>📈 Intelligent Stock Analysis & Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>EDA • Technicals • ML Forecasting</p>", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
stock = st.sidebar.text_input("Enter NSE Stock Symbol")

# =============================
# ANALYSIS
# =============================
if st.button("Analyze"):
    df, df1 = load_stock(stock)

    df = daily_return(df)
    monthly_df = monthly_return(df)
    yearly_df = yearly_return(df)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(show_return(df), use_container_width=True)
        st.plotly_chart(plot_daily_return(df), use_container_width=True)
        st.plotly_chart(plot_yearly_return(yearly_df), use_container_width=True)

    with col2:
        st.plotly_chart(comparison(df, df1), use_container_width=True)
        st.plotly_chart(plot_monthly_return(monthly_df), use_container_width=True)

        analysis = info(monthly_df)
        st.metric("Avg +ve Return", f"{analysis['avg_positive']:.2f}%")
        st.metric("Avg -ve Return", f"{analysis['avg_negative']:.2f}%")
        st.metric("Win Rate", f"{analysis['win_rate']:.2f}%")

        pe, sector = fundamental_info(stock)
        st.write("Sector:", sector)
        st.write("P/E:", pe)

# =============================
# TECHNICAL SIGNAL
# =============================
if st.button("Technical Signal"):
    df, _ = load_stock(stock)
    signal = ema_macd_signal(df)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Trend", signal["trend"])
    col2.metric("MACD", f"{signal['latest_macd']:.3f}")

    # Alert logic
    trend = signal["trend"].lower()
    macd = signal["latest_macd"]

    st.markdown("---")

    if ("bull" in trend or "up" in trend) and macd > 0:
        st.success("📈 **Bullish Signal Detected**\n\nEMA crossover + Positive MACD → BUY / HOLD")

    elif ("bear" in trend or "down" in trend) and macd < 0:
        st.error("📉 **Bearish Signal Detected**\n\nDowntrend + Negative MACD → SELL / AVOID")

    else:
        st.warning("⚠️ **No Clear Signal**\n\nIndicators are mixed → WAIT for confirmation")


# =============================
# ML PREDICTION
# =============================
if st.button("Predict Price"):
    df, _ = load_stock(stock)
    df = create_targets(create_features(df))

    features = ["return_1", "ema_ratio", "volatility", "Volume"]
    rf, xgb, *_ = train_models(df, features)

    pred = predict_next_day(df, rf, xgb, features)
    latest_close = df["Close"].iloc[-1]

    daily_ret = pred["return_pct"] / 100
    price_8 = predict_n_day_price(latest_close, daily_ret, 8)
    price_31 = predict_n_day_price(latest_close, daily_ret, 31)

    # 🔹 UI
    st.subheader("📊 Price Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Direction", pred["direction"])
    col2.metric("1-Day Price", f"₹{pred['predicted_price']:.2f}")
    col3.metric("1-Day Return", f"{pred['return_pct']:.2f}%")

    st.subheader("📅 Multi-Day Forecast")

    col4, col5 = st.columns(2)
    col4.metric(
        "📆 8-Day Price",
        f"₹{price_8:.2f}",
        f"{((price_8 - latest_close)/latest_close)*100:.2f}%"
    )

    col5.metric(
        "📅 31-Day Price",
        f"₹{price_31:.2f}",
        f"{((price_31 - latest_close)/latest_close)*100:.2f}%"
    )


# =============================
# NIFTY50 RECOMMENDER
# =============================
if st.button("Suggest NIFTY50"):
    stocks = get_nifty50_list()
    signals = scan_nifty_reversals(stocks)

    if signals.empty:
        st.info("No reversal signals today")
    else:
        st.subheader("📌 NIFTY50 Reversal Candidates")
        st.dataframe(signals, use_container_width=True)




