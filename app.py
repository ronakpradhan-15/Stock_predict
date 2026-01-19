import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ============================
# CONFIG
# ============================
MODEL_CURRENCY = "INR"
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

# ============================
# CSS
# ============================
st.markdown("""
<style>
.top-stock-row {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:4px 0;
    border-bottom:1px solid #2a2a2a;
}
.stock-left {display:flex;flex-direction:column;}
.stock-symbol {font-size:13px;font-weight:600;}
.stock-name {font-size:11px;color:#9aa0a6;}
.stock-right {text-align:right;font-size:13px;font-weight:600;}
.price-up {color:#00c853;}
.price-down {color:#ff5252;}
</style>
""", unsafe_allow_html=True)

# ============================
# TITLE
# ============================
st.title("📈 Stock Trend Prediction App")
st.write("Predict stock trend and visualize prices with live market data.")

# ============================
# LOAD MODEL
# ============================
model = joblib.load("stock_model.pkl")

# ============================
# CACHED TICKER (CRITICAL FIX)
# ============================
@st.cache_resource
def get_ticker(symbol):
    return yf.Ticker(symbol)

# ============================
# HELPERS
# ============================
@st.cache_data(ttl=600)
def get_exchange_rate(frm, to):
    if frm == to:
        return 1.0
    df = yf.Ticker(f"{frm}{to}=X").history(period="1d")
    return float(df["Close"].iloc[-1]) if not df.empty else 1.0


@st.cache_data(ttl=600)
def get_stock_data(symbol, time_range):
    period_map = {
        "1D": "1d","1W": "5d","1M": "1mo","3M": "3mo","6M": "6mo",
        "YTD": "ytd","1Y": "1y","2Y": "2y","5Y": "5y","10Y": "10y","ALL": "max"
    }
    ticker = get_ticker(symbol)
    df = ticker.history(period=period_map.get(time_range, "1mo"))
    currency = ticker.fast_info.get("currency", "USD")
    return df, currency


@st.cache_data(ttl=120)
def get_live_price(symbol):
    ticker = get_ticker(symbol)
    df = ticker.history(period="2d")
    if len(df) < 2:
        return None, None, ticker.fast_info.get("currency", "USD")

    price = float(df["Close"].iloc[-1])
    change = float(df["Close"].iloc[-1] - df["Close"].iloc[-2])
    currency = ticker.fast_info.get("currency", "USD")
    return price, change, currency


def plot_stock_chart(df, symbol, time_range, currency):
    up = df["Close"].iloc[-1] >= df["Close"].iloc[0]
    color = "green" if up else "red"
    fill = "rgba(0,255,0,0.25)" if up else "rgba(255,0,0,0.25)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=fill
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        title=f"{symbol} Price ({time_range})",
        yaxis_title=f"Price ({currency})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================
# LAYOUT
# ============================
main_col, side_col = st.columns([3, 1])

# ============================
# LEFT PANEL
# ============================
with main_col:
    st.subheader("🔍 Prediction")

    oc = st.number_input("Open – Close Difference", value=0.0)
    hl = st.number_input("High – Low Difference", value=0.0)

    input_currency = st.selectbox(
        "Prediction Currency", ["USD","INR","EUR","GBP"], index=1
    )

    if st.button("Predict"):
        rate = get_exchange_rate(input_currency, MODEL_CURRENCY)
        pred = model.predict(np.array([[oc * rate, hl * rate]]))[0]
        if pred == 1:
            st.success("📈 UP (Buy)")
        else:
            st.error("📉 DOWN (Sell)")

    st.divider()
    st.subheader("📊 Stock Chart")

    symbol = st.text_input("Stock Symbol", "AAPL")
    time_range = st.radio(
        "Time Range",
        ["1D","1W","1M","3M","6M","YTD","1Y","2Y","5Y","10Y","ALL"],
        horizontal=True,
        index=2
    )

    display_currency = st.selectbox(
        "Display Currency", ["USD","INR","EUR","GBP"], index=0
    )

    df, base_currency = get_stock_data(symbol, time_range)
    if not df.empty:
        rate = get_exchange_rate(base_currency, display_currency)
        df["Close"] *= rate
        plot_stock_chart(df, symbol, time_range, display_currency)
    else:
        st.warning("No data available")

# ============================
# RIGHT PANEL — TOP STOCKS
# ============================
TOP_STOCKS = [
    ("MRF.BO","MRF Ltd"),
    ("RELIANCE.NS","Reliance"),
    ("TCS.NS","TCS"),
    ("INFY.NS","Infosys"),
    ("HDFCBANK.NS","HDFC Bank"),
    ("SBIN.NS","SBI"),
    ("BEL.NS","Bharat Electronics"),
    ("ITC.NS","ITC")
]

with side_col:
    st.subheader("📌 Top Stocks")

    for sym, name in TOP_STOCKS:
        price, change, base_currency = get_live_price(sym)
        if price is None:
            continue

        rate = get_exchange_rate(base_currency, display_currency)
        price *= rate
        change *= rate

        cls = "price-up" if change >= 0 else "price-down"
        sign = "+" if change >= 0 else ""

        st.markdown(f"""
        <div class="top-stock-row">
            <div class="stock-left">
                <span class="stock-symbol">{sym}</span>
                <span class="stock-name">{name}</span>
            </div>
            <div class="stock-right {cls}">
                {price:,.2f}<br>{sign}{change:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
