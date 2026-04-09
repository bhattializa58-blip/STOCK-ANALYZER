"""
stock_analyzer.py — LSTM-Based Stock Investment Signal Analyzer
A complete Streamlit web application for educational purposes only.

Run with: streamlit run stock_analyzer.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import io

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

[data-testid="stSidebar"] {
    background: #0d1225 !important;
    border-right: 1px solid #1e2a45;
}

.signal-box {
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 16px 0;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}

.signal-strong-buy  { background: linear-gradient(135deg,#003d00,#006600); border: 2px solid #00cc44; color: #00ff66; }
.signal-buy         { background: linear-gradient(135deg,#002a00,#004d00); border: 2px solid #00aa33; color: #66ff99; }
.signal-hold        { background: linear-gradient(135deg,#2a2a00,#4d4400); border: 2px solid #aaaa00; color: #ffff66; }
.signal-sell        { background: linear-gradient(135deg,#2a0000,#550000); border: 2px solid #cc2200; color: #ff6644; }
.signal-strong-sell { background: linear-gradient(135deg,#1a0000,#440000); border: 2px solid #ff0000; color: #ff3333; }

.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #1a3a6e, #0d5c8a) !important;
    color: #a8d8ff !important;
    border: 1px solid #2a5fa0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 10px 20px !important;
    width: 100% !important;
}

.disclaimer {
    background: #1a1200;
    border: 1px solid #5a4500;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ccaa44;
    font-size: 0.82rem;
    margin-top: 24px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    color: #4a9eff;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "SPY  — S&P 500 ETF":          "SPY",
    "QQQ  — Nasdaq-100 ETF":       "QQQ",
    "DIA  — Dow Jones ETF":        "DIA",
    "IWM  — Russell 2000 ETF":     "IWM",
    "AAPL — Apple":                "AAPL",
    "MSFT — Microsoft":            "MSFT",
    "TSLA — Tesla":                "TSLA",
    "AMZN — Amazon":               "AMZN",
    "NVDA — NVIDIA":               "NVDA",
    "GLD  — Gold ETF":             "GLD",
    "XOM  — Exxon Mobil":          "XOM",
    "JPM  — JPMorgan Chase":       "JPM",
}

PERIODS = {"1 Year": 365, "2 Years": 730, "5 Years": 1825}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def download_data(ticker: str, days: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days + 90)   # extra buffer for MA warmup
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ── Candlestick body features ────────────────────────────────────────────
    d["body_size"]  = (d["Close"] - d["Open"]).abs()
    d["body_ratio"] = (d["Close"] - d["Open"]) / (d["High"] - d["Low"] + 1e-9)
    d["upper_shadow"]= d["High"] - d[["Close","Open"]].max(axis=1)
    d["lower_shadow"]= d[["Close","Open"]].min(axis=1) - d["Low"]
    d["direction"]  = np.sign(d["Close"] - d["Open"])

    # ── Return features ──────────────────────────────────────────────────────
    d["ret_1d"]  = d["Close"].pct_change(1)
    d["ret_5d"]  = d["Close"].pct_change(5)
    d["ret_10d"] = d["Close"].pct_change(10)

    # ── Trend features ───────────────────────────────────────────────────────
    d["sma20"] = d["Close"].rolling(20).mean()
    d["sma50"] = d["Close"].rolling(50).mean()
    d["price_to_sma20"] = d["Close"] / d["sma20"] - 1
    d["price_to_sma50"] = d["Close"] / d["sma50"] - 1

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ── Volatility (20-day rolling std of returns) ───────────────────────────
    d["volatility_20"] = d["ret_1d"].rolling(20).std()

    # ── Average True Range (ATR) ─────────────────────────────────────────────
    hl  = d["High"] - d["Low"]
    hcp = (d["High"] - d["Close"].shift(1)).abs()
    lcp = (d["Low"]  - d["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()

    # ── Volume relative to 20-day average ───────────────────────────────────
    d["vol_ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)

    # ── RSI (normalised to [0,1]) ────────────────────────────────────────────
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["rsi"] = (100 - 100 / (1 + rs)) / 100   # normalise to [0,1]

    # ── Bollinger Band width ─────────────────────────────────────────────────
    bb_std = d["Close"].rolling(20).std()
    bb_mid = d["sma20"]
    d["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)

    # ── %B (position within Bollinger Bands) ─────────────────────────────────
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d["bb_pct"] = (d["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    d.dropna(inplace=True)
    return d


def create_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """1 = BUY if Close rises > threshold% in `horizon` trading days, else 0."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > threshold / 100).astype(int)


FEATURE_COLS = [
    "body_size","body_ratio","upper_shadow","lower_shadow","direction",
    "ret_1d","ret_5d","ret_10d",
    "price_to_sma20","price_to_sma50",
    "macd","macd_signal","macd_hist",
    "volatility_20","atr_14","vol_ratio","rsi",
    "bb_width","bb_pct",
]


def prepare_data(df: pd.DataFrame, labels: pd.Series,
                 window: int, train_frac=0.80, val_frac=0.10):
    """Scale, window, and chronologically split data. Returns X, y arrays and scaler."""
    # Align labels with feature df
    valid_idx = labels.dropna().index
    feat = df[FEATURE_COLS].loc[valid_idx]
    lbl  = labels.loc[valid_idx]

    n = len(feat)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    scaler = MinMaxScaler()
    feat_arr = feat.values

    # Fit scaler ONLY on training portion
    scaler.fit(feat_arr[:train_end])
    feat_scaled = scaler.transform(feat_arr)

    lbl_arr = lbl.values

    def make_windows(X, y, start, end):
        Xs, ys = [], []
        for i in range(start, end):
            if i + window > len(X):
                break
            Xs.append(X[i:i + window])
            ys.append(y[i + window - 1])
        return np.array(Xs), np.array(ys)

    X_train, y_train = make_windows(feat_scaled, lbl_arr, 0, train_end)
    X_val,   y_val   = make_windows(feat_scaled, lbl_arr, train_end, val_end)
    X_test,  y_test  = make_windows(feat_scaled, lbl_arr, val_end, n)

    # Last window for live prediction
    X_live = feat_scaled[-window:][np.newaxis, :, :]

    split_info = {
        "train_samples": len(X_train), "val_samples": len(X_val),
        "test_samples": len(X_test),   "features": len(FEATURE_COLS),
        "window": window,
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, X_live, scaler, split_info


def build_model(window: int, n_features: int,
                units1: int, units2: int, dropout: float) -> tf.keras.Model:
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(window, n_features)),
        Dropout(dropout),
        LSTM(units2, return_sequences=False),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dropout(dropout / 2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def map_signal(prob: float):
    if prob >= 0.65:
        return "STRONG BUY",  "signal-strong-buy",  "🚀", "#00ff66"
    elif prob >= 0.52:
        return "BUY",         "signal-buy",         "📈", "#66ff99"
    elif prob >= 0.35:
        return "HOLD / WAIT", "signal-hold",        "⏸️", "#ffff66"
    elif prob >= 0.20:
        return "SELL",        "signal-sell",        "📉", "#ff6644"
    else:
        return "STRONG SELL", "signal-strong-sell", "🔻", "#ff3333"


def explain_signal(signal: str, prob: float, ticker: str,
                   df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    rsi_pct = latest["rsi"] * 100
    vol_note = "above-average" if latest["vol_ratio"] > 1.2 else \
               "below-average" if latest["vol_ratio"] < 0.8 else "normal"
    trend = "above" if latest["price_to_sma20"] > 0 else "below"

    explanations = {
        "STRONG BUY": (
            f"The LSTM model assigns a {prob:.1%} probability of a significant "
            f"upward move for **{ticker}**. The price is currently {trend} its 20-day moving average, "
            f"RSI sits at {rsi_pct:.1f} (not overbought), and volume is {vol_note}. "
            f"All signals point toward bullish momentum in the near-to-mid term."
        ),
        "BUY": (
            f"The model sees a {prob:.1%} likelihood of upward price movement for **{ticker}**. "
            f"Trend and momentum indicators lean positive. RSI is at {rsi_pct:.1f}. "
            f"Consider entering a position with appropriate risk management."
        ),
        "HOLD / WAIT": (
            f"The model is uncertain ({prob:.1%} BUY probability) about **{ticker}**'s near-term direction. "
            f"The price is {trend} its 20-day SMA and RSI stands at {rsi_pct:.1f}. "
            f"Neither a clear buy nor sell signal is present — waiting for more decisive confirmation is prudent."
        ),
        "SELL": (
            f"The model gives only a {prob:.1%} probability of upside for **{ticker}**. "
            f"Price is {trend} the 20-day SMA with RSI at {rsi_pct:.1f}. "
            f"Momentum appears to be weakening; consider reducing exposure."
        ),
        "STRONG SELL": (
            f"The model assigns just {prob:.1%} probability of a rally for **{ticker}**. "
            f"Multiple indicators suggest bearish pressure: the price is {trend} its 20-day SMA, "
            f"RSI is at {rsi_pct:.1f}, and volume is {vol_note}. "
            f"The pattern sequence strongly resembles historical downward moves."
        ),
    }
    return explanations.get(signal, "")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 LSTM Stock Analyzer")
    st.markdown("---")

    ticker_label = st.selectbox("**Ticker**", list(TICKERS.keys()), index=0)
    ticker = TICKERS[ticker_label]

    period_label = st.selectbox("**Historical Period**", list(PERIODS.keys()), index=1)
    period_days  = PERIODS[period_label]

    st.markdown("---")
    st.markdown("**Model Hyperparameters**")

    window   = st.slider("Lookback Window (days)",  10, 60, 30, 5)
    horizon  = st.slider("Forecast Horizon (days)",  3, 30, 10, 1)
    threshold= st.slider("BUY Threshold (%)",       0.5, 5.0, 2.0, 0.5)
    units1   = st.select_slider("LSTM Layer 1 Units", [32, 64, 128, 256], value=128)
    units2   = st.select_slider("LSTM Layer 2 Units", [16, 32, 64, 128], value=64)
    dropout  = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)

    st.markdown("---")
    run = st.button("▶  RUN ANALYSIS")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"# 📈 LSTM Stock Signal Analyzer")
st.markdown(
    f"**Ticker:** `{ticker}` &nbsp;|&nbsp; "
    f"**Period:** {period_label} &nbsp;|&nbsp; "
    f"**Window:** {window}d &nbsp;|&nbsp; "
    f"**Horizon:** {horizon}d &nbsp;|&nbsp; "
    f"**Threshold:** {threshold}%"
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE (only runs on button click)
# ─────────────────────────────────────────────────────────────────────────────
if not run:
    st.info("👈  Configure your parameters in the sidebar and click **▶ RUN ANALYSIS** to begin.")

    # Show sample tuning guide even before running
    st.markdown('<div class="section-header">⚙️ Hyperparameter Tuning Guide</div>', unsafe_allow_html=True)
    guide = pd.DataFrame({
        "Parameter":    ["Window",  "Horizon", "BUY Threshold", "LSTM Units", "Dropout"],
        "Recommended":  ["20–40",   "5–15",    "1.5–3.0%",      "64–128",    "0.2–0.3"],
        "Effect (↑)":   [
            "More context, slower training",
            "Longer forecast, harder task",
            "Fewer BUY labels (rarer signal)",
            "More capacity, risk of overfit",
            "Less overfit, may underfit",
        ],
        "Effect (↓)":   [
            "Less context, faster training",
            "Short-term focus, noisier",
            "More BUY labels (loose signal)",
            "Less capacity, may underfit",
            "More overfit, faster training",
        ],
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)
    st.stop()

# ── Step 1: Download ─────────────────────────────────────────────────────────
with st.spinner(f"Downloading {ticker} data…"):
    raw_df = download_data(ticker, period_days)

if raw_df.empty or len(raw_df) < 100:
    st.error("Not enough data returned. Try a longer period or different ticker.")
    st.stop()

# ── Step 2: Feature Engineering ─────────────────────────────────────────────
with st.spinner("Engineering features…"):
    feat_df = engineer_features(raw_df)

# ── Candlestick Chart ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🕯️ Price Chart</div>', unsafe_allow_html=True)

fig_candle = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.75, 0.25], vertical_spacing=0.03,
)

fig_candle.add_trace(go.Candlestick(
    x=feat_df.index, open=feat_df["Open"], high=feat_df["High"],
    low=feat_df["Low"], close=feat_df["Close"],
    name="OHLC", increasing_line_color="#00cc44", decreasing_line_color="#ff4444",
), row=1, col=1)

fig_candle.add_trace(go.Scatter(
    x=feat_df.index, y=feat_df["sma20"], name="SMA 20",
    line=dict(color="#4a9eff", width=1.5),
), row=1, col=1)

fig_candle.add_trace(go.Scatter(
    x=feat_df.index, y=feat_df["sma50"], name="SMA 50",
    line=dict(color="#ff9f4a", width=1.5),
), row=1, col=1)

fig_candle.add_trace(go.Bar(
    x=feat_df.index, y=feat_df["Volume"], name="Volume",
    marker_color="#2a4a7f", opacity=0.7,
), row=2, col=1)

fig_candle.update_layout(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
    font_color="#e0e6f0", height=480,
    xaxis_rangeslider_visible=False,
    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=0, r=0, t=10, b=0),
)
fig_candle.update_xaxes(gridcolor="#1e2a45", showgrid=True)
fig_candle.update_yaxes(gridcolor="#1e2a45", showgrid=True)

st.plotly_chart(fig_candle, use_container_width=True)

# ── Feature Table (expandable) ────────────────────────────────────────────────
with st.expander("📋 Feature Table (last 20 rows)"):
    display_cols = FEATURE_COLS + ["Close"]
    st.dataframe(feat_df[display_cols].tail(20).round(4), use_container_width=True)

# ── Step 3: Labels & Data Preparation ────────────────────────────────────────
with st.spinner("Preparing data & creating labels…"):
    labels = create_labels(feat_df, horizon, threshold)
    result = prepare_data(feat_df, labels, window)
    X_train, y_train, X_val, y_val, X_test, y_test, X_live, scaler, split_info = result

# Data split statistics
st.markdown('<div class="section-header">📂 Data Split Statistics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train Samples",    split_info["train_samples"])
c2.metric("Validation Samples", split_info["val_samples"])
c3.metric("Test Samples",     split_info["test_samples"])
c4.metric("Features",         split_info["features"])

col_a, col_b = st.columns(2)
buy_pct_train = y_train.mean() * 100
buy_pct_test  = y_test.mean()  * 100
col_a.metric("BUY % in Training Set", f"{buy_pct_train:.1f}%")
col_b.metric("BUY % in Test Set",     f"{buy_pct_test:.1f}%")

if split_info["train_samples"] < 50:
    st.warning("⚠️ Very few training samples. Try a longer period or smaller window.")
    st.stop()

# ── Step 4: Build Model ───────────────────────────────────────────────────────
model = build_model(window, len(FEATURE_COLS), units1, units2, dropout)

# Model summary (expandable)
with st.expander("🏗️ Model Architecture Summary"):
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    st.code(buf.getvalue(), language="text")

# ── Step 5: Training ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 Model Training</div>', unsafe_allow_html=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=6, min_lr=1e-6, verbose=0),
]

progress_bar = st.progress(0, text="Training LSTM…")
status_text  = st.empty()

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total = total_epochs
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss", 0))
        self.history["val_loss"].append(logs.get("val_loss", 0))
        self.history["accuracy"].append(logs.get("accuracy", 0))
        self.history["val_accuracy"].append(logs.get("val_accuracy", 0))
        pct = min((epoch + 1) / self.total, 1.0)
        progress_bar.progress(pct,
            text=f"Epoch {epoch+1} | loss: {logs.get('loss',0):.4f}  "
                 f"val_loss: {logs.get('val_loss',0):.4f}  "
                 f"acc: {logs.get('accuracy',0):.4f}")

MAX_EPOCHS = 100
st_cb = StreamlitCallback(MAX_EPOCHS)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=32,
    callbacks=callbacks + [st_cb],
    verbose=0,
)

progress_bar.empty()
status_text.success(f"✅ Training complete — {len(st_cb.history['loss'])} epochs")

# Loss curve
epochs_ran = list(range(1, len(st_cb.history["loss"]) + 1))
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=epochs_ran, y=st_cb.history["loss"],
                              name="Train Loss", line=dict(color="#4a9eff", width=2)))
fig_loss.add_trace(go.Scatter(x=epochs_ran, y=st_cb.history["val_loss"],
                              name="Val Loss", line=dict(color="#ff9f4a", width=2)))
fig_loss.add_trace(go.Scatter(x=epochs_ran, y=st_cb.history["accuracy"],
                              name="Train Acc", line=dict(color="#44ff88", width=1.5, dash="dot")))
fig_loss.add_trace(go.Scatter(x=epochs_ran, y=st_cb.history["val_accuracy"],
                              name="Val Acc", line=dict(color="#ffdd44", width=1.5, dash="dot")))
fig_loss.update_layout(
    title="Training Loss & Accuracy",
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
    font_color="#e0e6f0", height=300,
    xaxis_title="Epoch", yaxis_title="Value",
    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig_loss.update_xaxes(gridcolor="#1e2a45")
fig_loss.update_yaxes(gridcolor="#1e2a45")
st.plotly_chart(fig_loss, use_container_width=True)

# ── Step 6: Evaluation ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Evaluation</div>', unsafe_allow_html=True)

y_prob_test = model.predict(X_test, verbose=0).flatten()
y_pred_test = (y_prob_test >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred_test)
try:
    auc = roc_auc_score(y_test, y_prob_test)
except Exception:
    auc = float("nan")

report_dict = classification_report(y_test, y_pred_test,
                                    target_names=["SELL/HOLD", "BUY"],
                                    output_dict=True, zero_division=0)
report_text = classification_report(y_test, y_pred_test,
                                    target_names=["SELL/HOLD", "BUY"],
                                    zero_division=0)

m1, m2, m3 = st.columns(3)
m1.metric("Test Accuracy", f"{acc:.3f}")
m2.metric("ROC-AUC",       f"{auc:.3f}" if not np.isnan(auc) else "N/A")
m3.metric("Test Samples",  len(y_test))

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Classification Report**")
    st.code(report_text, language="text")

with col2:
    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted SELL/HOLD", "Predicted BUY"],
        y=["Actual SELL/HOLD", "Actual BUY"],
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e0e6f0", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── BUY Probability chart over test period ────────────────────────────────────
st.markdown('<div class="section-header">📉 BUY Probability — Test Period</div>', unsafe_allow_html=True)

# Map test predictions back to dates
test_start_row = split_info["train_samples"] + split_info["val_samples"] + window - 1
feat_valid_idx = feat_df.index[labels.notna()]
test_dates = feat_valid_idx[test_start_row: test_start_row + len(y_prob_test)]

fig_prob = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.6, 0.4], vertical_spacing=0.04)

fig_prob.add_trace(go.Scatter(
    x=test_dates, y=y_prob_test, name="BUY Probability",
    line=dict(color="#4a9eff", width=1.5), fill="tozeroy",
    fillcolor="rgba(74,158,255,0.15)",
), row=1, col=1)

# Thresholds
for lvl, col, nm in [(0.65, "#00ff66", "Strong Buy"), (0.52, "#66ff99", "Buy"),
                      (0.35, "#ffff66", "Hold"),       (0.20, "#ff6644", "Sell")]:
    fig_prob.add_hline(y=lvl, line_dash="dash", line_color=col,
                        annotation_text=nm, annotation_position="right",
                        row=1, col=1)

# Price over test period
if len(test_dates) > 0:
    test_prices = feat_df["Close"].reindex(test_dates)
    fig_prob.add_trace(go.Scatter(
        x=test_dates, y=test_prices, name="Close Price",
        line=dict(color="#e0e6f0", width=1.5),
    ), row=2, col=1)

    # Overlay signals
    buy_mask  = y_pred_test == 1
    sell_mask = y_pred_test == 0
    fig_prob.add_trace(go.Scatter(
        x=test_dates[buy_mask],  y=test_prices[buy_mask],
        mode="markers", name="BUY signal",
        marker=dict(color="#00ff66", size=6, symbol="triangle-up"),
    ), row=2, col=1)
    fig_prob.add_trace(go.Scatter(
        x=test_dates[sell_mask], y=test_prices[sell_mask],
        mode="markers", name="SELL/HOLD signal",
        marker=dict(color="#ff4444", size=6, symbol="triangle-down"),
    ), row=2, col=1)

fig_prob.update_layout(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
    font_color="#e0e6f0", height=420,
    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=0, r=0, t=10, b=0),
)
fig_prob.update_xaxes(gridcolor="#1e2a45")
fig_prob.update_yaxes(gridcolor="#1e2a45")
st.plotly_chart(fig_prob, use_container_width=True)

# ── LIVE PREDICTION ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Investment Signal</div>', unsafe_allow_html=True)

live_prob = float(model.predict(X_live, verbose=0)[0][0])
signal, css_class, icon, sig_color = map_signal(live_prob)
explanation = explain_signal(signal, live_prob, ticker, feat_df)

st.markdown(
    f'<div class="signal-box {css_class}">'
    f'{icon}&nbsp;&nbsp;{signal}&nbsp;&nbsp;{icon}<br>'
    f'<span style="font-size:1rem;opacity:0.8">Probability: {live_prob:.1%}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(f"**{signal} Explanation**")
st.markdown(explanation)

# Latest metrics table
st.markdown("**Latest Technical Indicators**")
latest = feat_df.iloc[-1]
ind_data = {
    "RSI":             f"{latest['rsi']*100:.1f}",
    "MACD":            f"{latest['macd']:.4f}",
    "Volume Ratio":    f"{latest['vol_ratio']:.2f}x",
    "vs SMA-20":       f"{latest['price_to_sma20']*100:+.2f}%",
    "vs SMA-50":       f"{latest['price_to_sma50']*100:+.2f}%",
    "BB Width":        f"{latest['bb_width']:.4f}",
    "BB %B":           f"{latest['bb_pct']:.3f}",
    "20d Volatility":  f"{latest['volatility_20']*100:.2f}%",
}
ind_df = pd.DataFrame(ind_data.items(), columns=["Indicator", "Value"])
st.dataframe(ind_df, use_container_width=True, hide_index=True)

# ── Tuning Guide ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Hyperparameter Tuning Guide</div>', unsafe_allow_html=True)
guide = pd.DataFrame({
    "Parameter":    ["Window",  "Horizon", "BUY Threshold", "LSTM Units", "Dropout"],
    "Current":      [str(window), str(horizon), f"{threshold}%",
                     f"{units1}/{units2}", str(dropout)],
    "Recommended":  ["20–40",   "5–15",    "1.5–3.0%",      "64–128",    "0.2–0.3"],
    "Effect (↑)":   [
        "More context, slower training",
        "Longer forecast, harder task",
        "Fewer BUY labels (rarer signal)",
        "More capacity, risk of overfit",
        "Less overfit, may underfit",
    ],
})
st.dataframe(guide, use_container_width=True, hide_index=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <strong>EDUCATIONAL PURPOSES ONLY</strong> — This application is a machine-learning '
    'exercise for academic use. The output is NOT financial advice. Past patterns '
    'do not guarantee future results. Never invest money based on model outputs alone. '
    'Always consult a qualified financial advisor before making investment decisions.'
    '</div>',
    unsafe_allow_html=True,
)
