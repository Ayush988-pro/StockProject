
# =============================
# IMPORTS
# =============================
import streamlit as st

from loading import (
    load_stock,
    show_return,
    comparison,
    daily_return,
    plot_daily_return,
    monthly_return,
    plot_monthly_return,
    yearly_return,
    plot_yearly_return,
    info,
    fundamental_info,
    ema_macd_signal,
    create_features,
    create_targets,
    train_models,
    predict_next_day,
    predict_n_day_price,
    analyze_news_sentiment_finbert,
    get_nifty50_list,
    scan_nifty_reversals,
)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide",
)

# =============================
# GLOBAL STYLES
# =============================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    h1, h2, h3, h4 {
        color: #4da3ff;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 10px;
    }
    div.stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("Stock Dashboard")

stock = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    placeholder="e.g. RELIANCE or TCS",
).strip()

st.sidebar.caption(
    "Enter the NSE symbol without .NS, or include .NS if preferred."
)

# =============================
# MARKET SENTIMENT (FINBERT)
# =============================
st.subheader("📰 Market Sentiment")

news_urls = [
    "https://www.reuters.com/world/india/indian-shares-fall-as-it-stocks-drag-markets-2024-01-07/",
    "https://www.moneycontrol.com/news/business/markets/sensex-falls-over-400-points-nifty-below-26300-it-stocks-drag-106640997.html",
]

color_map = {
    "Bullish 🟢": "#00c853",
    "Bearish 🔴": "#ff5252",
    "Neutral ⚪": "#ffca28",
}

# Avoid blocking every app interaction with news downloads.
# The sentiment analysis runs only when requested.
if st.button("Refresh Market Sentiment"):
    with st.spinner("Analyzing financial news with FinBERT..."):
        try:
            sentiment, score = analyze_news_sentiment_finbert(
                news_urls
            )

            color = color_map.get(sentiment, "#ffffff")

            st.markdown(
                f"""
                <div style="
                    background:#111827;
                    padding:14px;
                    border:1px solid #333;
                    border-radius:8px;
                    color:{color};
                    font-weight:600;
                ">
                    Market Sentiment: {sentiment}
                    <br>
                    Model Confidence: {score:.2%}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(
                "Unable to analyze market sentiment. "
                "Check the app logs and dependencies."
            )
            print(f"FinBERT error: {exc}")

# =============================
# TITLE
# =============================
st.markdown(
    """
    <h1 style="text-align:center;">
        📈 Intelligent Stock Analysis & Prediction
    </h1>
    <p style="text-align:center;color:gray;">
        EDA • Technical Indicators • ML Forecasting
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# =============================
# HELPER: LOAD VALID STOCK
# =============================
def get_stock_data():
    if not stock:
        st.warning("Please enter an NSE stock symbol in the sidebar.")
        return None, None

    try:
        with st.spinner(f"Downloading data for {stock.upper()}..."):
            return load_stock(stock)

    except Exception as exc:
        st.error(
            "Unable to load stock data. Verify the symbol and "
            "check whether Yahoo Finance is available."
        )
        print(f"Stock loading error: {exc}")
        return None, None


# =============================
# ANALYSIS
# =============================
if st.button("Analyze", type="primary"):
    df, df1 = get_stock_data()

    if df is not None and df1 is not None:
        try:
            df = daily_return(df)

            monthly_df = monthly_return(df)
            yearly_df = yearly_return(df)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Stock Price")
                st.plotly_chart(
                    show_return(df),
                    use_container_width=True,
                )

                st.subheader("📉 Daily Returns")
                st.plotly_chart(
                    plot_daily_return(df),
                    use_container_width=True,
                )

                st.subheader("📅 Yearly Returns")
                st.plotly_chart(
                    plot_yearly_return(yearly_df),
                    use_container_width=True,
                )

            with col2:
                st.subheader("📈 Stock vs NIFTY 50")
                st.plotly_chart(
                    comparison(df, df1),
                    use_container_width=True,
                )

                st.subheader("📆 Monthly Returns")
                st.plotly_chart(
                    plot_monthly_return(monthly_df),
                    use_container_width=True,
                )

                analysis = info(monthly_df)

                st.subheader("📌 Return Statistics")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Avg Positive Return",
                    f"{analysis['avg_positive']:.2f}%",
                )

                m2.metric(
                    "Avg Negative Return",
                    f"{analysis['avg_negative']:.2f}%",
                )

                m3.metric(
                    "Win Rate",
                    f"{analysis['win_rate']:.2f}%",
                )

                st.subheader("🏢 Fundamentals")

                pe, sector = fundamental_info(stock)

                st.write(
                    "Sector:",
                    sector if sector else "Unavailable",
                )

                st.write(
                    "Trailing P/E:",
                    f"{pe:.2f}" if pe is not None else "Unavailable",
                )

        except Exception as exc:
            st.error(
                "An error occurred while generating the analysis. "
                "Check the app logs for details."
            )
            print(f"Analysis error: {exc}")


# =============================
# TECHNICAL SIGNAL
# =============================
if st.button("Technical Signal"):
    df, _ = get_stock_data()

    if df is not None:
        try:
            signal = ema_macd_signal(df)

            col1, col2 = st.columns(2)

            col1.metric("Long-Term Trend", signal["trend"])
            col2.metric(
                "MACD",
                f"{signal['latest_macd']:.3f}",
            )

            trend = signal["trend"].lower()
            macd = signal["latest_macd"]

            st.divider()

            if "bull" in trend and macd > 0:
                st.success(
                    "📈 Bullish technical conditions detected. "
                    "Confirm with other indicators before making "
                    "an investment decision."
                )

            elif "bear" in trend and macd < 0:
                st.error(
                    "📉 Bearish technical conditions detected. "
                    "Review risk and other indicators before acting."
                )

            else:
                st.warning(
                    "⚠️ No clear combined signal. "
                    "The indicators may be mixed."
                )

            st.caption(
                "The trend uses EMA50 versus EMA200. "
                "MACD is calculated using EMA12 minus EMA26."
            )

        except Exception as exc:
            st.error("Unable to calculate technical indicators.")
            print(f"Technical analysis error: {exc}")


# =============================
# ML PRICE PREDICTION
# =============================
if st.button("Predict Price"):
    df, _ = get_stock_data()

    if df is not None:
        try:
            features = [
                "return_1",
                "ema_ratio",
                "volatility",
                "Volume",
            ]

            # Features for all valid rows, including the latest row.
            feature_df = create_features(df.copy())

            if feature_df.empty:
                st.error("Insufficient data to create ML features.")
                st.stop()

            # Training data must have a known next-day target.
            training_df = create_targets(feature_df.copy())

            if training_df.empty:
                st.error("Insufficient data to train the models.")
                st.stop()

            with st.spinner("Training ML models..."):
                rf, xgb, *_ = train_models(
                    training_df,
                    features,
                )

                # Predict using the latest available feature row,
                # not the previous row with a known target.
                pred = predict_next_day(
                    feature_df,
                    rf,
                    xgb,
                    features,
                )

            latest_close = float(feature_df["Close"].iloc[-1])

            daily_ret = pred["return_pct"] / 100

            price_8 = predict_n_day_price(
                latest_close,
                daily_ret,
                8,
            )

            price_31 = predict_n_day_price(
                latest_close,
                daily_ret,
                31,
            )

            st.subheader("📊 Next-Day Price Estimate")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Direction",
                pred["direction"],
            )

            col2.metric(
                "Estimated 1-Day Price",
                f"₹{pred['predicted_price']:.2f}",
            )

            col3.metric(
                "Estimated 1-Day Return",
                f"{pred['return_pct']:.2f}%",
            )

            st.subheader("📅 Illustrative Multi-Day Scenarios")

            col4, col5 = st.columns(2)

            col4.metric(
                "8-Day Price Scenario",
                f"₹{price_8:.2f}",
                f"{((price_8 / latest_close) - 1) * 100:.2f}%",
            )

            col5.metric(
                "31-Day Price Scenario",
                f"₹{price_31:.2f}",
                f"{((price_31 / latest_close) - 1) * 100:.2f}%",
            )

            st.caption(
                "Estimates are based on historical data and a "
                "machine-learning model. Multi-day scenarios assume "
                "a constant daily return and are not reliable forecasts "
                "of future market prices."
            )

        except Exception as exc:
            st.error(
                "Price prediction failed. Check the app logs "
                "for the underlying error."
            )
            print(f"Prediction error: {exc}")


# =============================
# NIFTY50 REVERSAL SCANNER
# =============================
if st.button("Suggest NIFTY50"):
    try:
        stocks = get_nifty50_list()

        with st.spinner("Scanning stocks for potential reversals..."):
            signals = scan_nifty_reversals(stocks)

        st.subheader("📌 NIFTY50 Reversal Candidates")

        if signals.empty:
            st.info(
                "No stocks in the configured list currently meet "
                "the reversal conditions."
            )
        else:
            st.dataframe(
                signals,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Scanned the {len(stocks)} stocks configured in the "
            "current list. Signals are technical screening results, "
            "not guaranteed buy recommendations."
        )

    except Exception as exc:
        st.error("The reversal scanner failed. Check the app logs.")
        print(f"Scanner error: {exc}")





