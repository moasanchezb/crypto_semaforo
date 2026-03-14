# app_semaforo.py
# Streamlit dashboard semáforo multi-cripto (15m)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import math
from datetime import datetime

st.set_page_config(page_title="Semáforo Crypto - Live", layout="wide")

# Auto refresh
st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

# ---------------- Config
MARKETS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "OPUSDT",
    "ARBUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "FTMUSDT",
    "ALGOUSDT",
    "SANDUSDT",
    "FILUSDT",
]

INTERVAL_RAW = "1m"
RESAMPLE_RULE = "15min"
FETCH_LIMIT = 1000

# Parámetros semáforo
EMA_SHORT = 100
EMA_LONG  = 300
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 60
ATR_PERIOD = 14
ATR_MULT = 2.5

WEIGHT_EMA = 0.3
WEIGHT_RSI = 0.4
WEIGHT_ATR = 0.3

SCORE_THRESH_LONG = 0.6
SCORE_THRESH_SHORT = -0.6

ENDPOINTS = [
    "https://data-api.binance.vision"
]

os.makedirs("outputs", exist_ok=True)

# ---------------- Utilities

@st.cache_data(ttl=15)
def fetch_klines_latest(symbol, interval="1m", limit=500):
    last_err = None
    for base in ENDPOINTS:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "streamlit-semaforo"})
            if r.status_code == 200:
                data = r.json()
                df = pd.DataFrame(data, columns=[
                    "openTime","open","high","low","close","volume","closeTime",
                    "quoteVol","trades","takerBase","takerQuote","ignore"
                ])
                df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
                for c in ["open","high","low","close","volume","quoteVol","trades"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"All endpoints failed. Last: {last_err}")


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def resample_to_15m(df):
    df = df.set_index("ts").sort_index()
    o = df["open"].resample(RESAMPLE_RULE).first()
    h = df["high"].resample(RESAMPLE_RULE).max()
    l = df["low"].resample(RESAMPLE_RULE).min()
    c = df["close"].resample(RESAMPLE_RULE).last()
    v = df["volume"].resample(RESAMPLE_RULE).sum()
    out = pd.concat([o,h,l,c,v], axis=1).dropna()
    out.columns = ["open","high","low","close","volume"]
    return out.reset_index()

def compute_semaforo(df15):
    df = df15.copy()
    df["EMA_short"] = ema(df["close"], EMA_SHORT)
    df["EMA_long"]  = ema(df["close"], EMA_LONG)
    df["EMA_slope"] = df["EMA_short"].diff()
    df["RSI"]       = rsi(df["close"], RSI_PERIOD)
    df["ATR"]       = atr(df, ATR_PERIOD)

    scores = []
    signals = []

    for _, row in df.iterrows():
        score = 0

        # EMA
        if row["EMA_short"] > row["EMA_long"] and row["EMA_slope"] > 0:
            score += WEIGHT_EMA
        elif row["EMA_short"] < row["EMA_long"] and row["EMA_slope"] < 0:
            score -= WEIGHT_EMA

        # RSI
        if row["RSI"] < RSI_OVERSOLD:
            score += WEIGHT_RSI
        elif row["RSI"] > RSI_OVERBOUGHT:
            score -= WEIGHT_RSI

        # ATR
        if row["ATR"] < ATR_MULT:
            score += WEIGHT_ATR
        else:
            score -= WEIGHT_ATR

        scores.append(score)

        if score >= SCORE_THRESH_LONG:
            signals.append("BUY")
        elif score <= SCORE_THRESH_SHORT:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    df["score"]  = scores
    df["signal"] = signals
    return df


# ===============================================================
#                       UI: MULTI-MERCADO
# ===============================================================

st.title("🚦 Semáforo Crypto — Multi-Mercado (15m)")

bankroll = st.number_input("Bankroll (USD)", value=34.77, min_value=1.0)

# Procesar cada mercado
for i in range(0, len(MARKETS), 2):
    cols = st.columns(2)

    for idx, col in enumerate(cols):
        if i + idx >= len(MARKETS):
            break

        symbol = MARKETS[i + idx]

        with col:
            st.markdown("---")
            st.header(f"{symbol}")

            try:
                # Fetch
                df1m = fetch_klines_latest(symbol, INTERVAL_RAW, FETCH_LIMIT)
                df15 = resample_to_15m(df1m)
                df_sig = compute_semaforo(df15)
                last = df_sig.iloc[-1]

                price = last["close"]
                score = float(last["score"])
                signal = last["signal"]
                timestamp = df15["ts"].iloc[-1]

                # Kelly
                p = 1 / (1 + math.exp(-score))
                b = 1.2
                kelly = max(0, (b*p - (1-p)) / b)
                stake = bankroll * kelly

                # Colors
                color = {"BUY": "#00C853", "SELL": "#D50000"}.get(signal, "#FFD43B")

                # Main box
                st.markdown(
                    f"""
                    <div style='background:{color};padding:25px;border-radius:12px;text-align:center;'>
                        <h1 style='color:white;margin:0;'>{signal}</h1>
                        <h3 style='color:white;margin:0;'>{symbol} {price:.4f} USD</h3>
                        <p style='color:white;margin:0;'>Score: {score:.3f} • Time: {timestamp.strftime("%Y-%m-%d %H:%M UTC")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Kelly box
                st.write(f"Probabilidad estimada: **{p:.3f}**")
                st.write(f"Fracción Kelly: **{kelly:.4f}**")
                st.write(f"Monto recomendado: **{stake:.2f} USD**")
                st.write(f"Quarter Kelly: **{stake * 0.25:.2f} USD**")

                # Table
                with st.expander("Tabla de indicadores (últimos 20)"):
                    st.table(df_sig.tail(20))

                # Chart
                st.line_chart(df_sig.set_index('ts')[['close']].tail(200))

            except Exception as e:
                st.error(f"Error en {symbol}: {e}")

