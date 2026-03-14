#!/usr/bin/env python3
"""
main.py - pipeline robusto para 90 días de 1m klines + semáforo (MA10/MA30 + RSI14 + MACD)
Reemplaza tu main.py anterior por este. Está pensado para macOS con Python 3.14 en venv.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np

# ---------------- setup / logging ----------------
def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

# ---------------- config load ----------------
with open("config.json", "r") as f:
    CFG = json.load(f)

DATA_FOLDER = CFG.get("data_folder", "data")
OUT_FOLDER = CFG.get("output_folder", "outputs")
LOG_FILE = CFG.get("log_file", "logs/app.log")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

setup_logging(LOG_FILE)

SYMBOL = CFG.get("symbol", "BTCUSDT")
INTERVAL = CFG.get("interval_raw", "1m")
ENDPOINTS = CFG.get("endpoints", ["https://api.binance.com"])
FETCH_RETRIES = int(CFG.get("fetch_retries", 2))

# interval to ms
INTERVAL_MS_MAP = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000
}
INTERVAL_MS = INTERVAL_MS_MAP.get(INTERVAL, 60_000)

# ---------------- helpers: fetch with rotation & pagination ----------------
def _fetch_url(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"python-requests/crypto-semaforo"})
        return r
    except Exception as e:
        logging.exception("Request exception: %s", e)
        return None

def fetch_klines_paginated(symbol: str, interval: str, days: int = 90, endpoints: List[str] = None, limit_per_call: int = 1000) -> pd.DataFrame:
    """
    Downloads 'days' worth of interval klines from Binance using pagination via startTime.
    Returns DataFrame with columns: ts, open, high, low, close, volume, closeTime, quoteAssetVolume, trades
    """
    endpoints = endpoints or ENDPOINTS
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_dt = datetime.now(timezone.utc) - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    all_rows = []
    attempt_info = []

    # We request blocks starting at start_ms and increase by limit*interval_ms each iteration
    cur_start = start_ms
    final_end = now_ms

    logging.info("Starting paginated fetch from %s to %s (days=%d)", datetime.fromtimestamp(start_ms/1000, timezone.utc), datetime.fromtimestamp(final_end/1000, timezone.utc), days)

    while cur_start < final_end:
        fetched = False
        last_err = None
        for base in endpoints:
            url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur_start}&limit={limit_per_call}"
            for attempt in range(1, FETCH_RETRIES + 1):
                r = _fetch_url(url)
                if r is None:
                    last_err = f"no response from {base}"
                    time.sleep(0.5)
                    continue
                code = r.status_code
                if code == 200:
                    try:
                        data = r.json()
                        if not isinstance(data, list):
                            last_err = f"unexpected structure from {base}"
                            logging.warning(last_err)
                            time.sleep(0.3)
                            continue
                        if len(data) == 0:
                            # no more data
                            fetched = True
                            cur_start = final_end
                            break
                        # Append mapped rows
                        for k in data:
                            # k: [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, trades, ...]
                            all_rows.append([
                                int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                                float(k[5]), int(k[6]), float(k[7]) if k[7] is not None else 0.0, int(k[8]) if k[8] is not None else 0
                            ])
                        # Move start to last returned openTime + interval_ms
                        last_open = int(data[-1][0])
                        cur_start = last_open + INTERVAL_MS
                        fetched = True
                        logging.info("Fetched %d bars from %s (last open %s)", len(data), base, datetime.fromtimestamp(last_open/1000, timezone.utc).isoformat())
                        # small sleep to be polite
                        time.sleep(0.12)
                        break
                    except Exception as e:
                        last_err = f"parse error {e}"
                        logging.exception(last_err)
                        time.sleep(0.3)
                        continue
                else:
                    last_err = f"{base} returned {code}"
                    logging.warning("Endpoint %s returned %s", base, code)
                    # backoff a bit on errors like 429/5xx
                    time.sleep(0.4 * attempt)
                    continue
            if fetched:
                break
        if not fetched:
            # none endpoint worked this loop -> abort to avoid infinite loop
            raise RuntimeError(f"Failed to fetch block starting at {datetime.fromtimestamp(cur_start/1000, timezone.utc)}. LastErr: {last_err}")
        # safety: don't loop forever
        if len(all_rows) > (days * 24 * 60 * 2):  # twice expected, sanity cap
            logging.warning("Too many rows accumulated, breaking to avoid runaway")
            break

    # Build DataFrame
    df = pd.DataFrame(all_rows, columns=["openTime","open","high","low","close","volume","closeTime","quoteAssetVolume","trades"])
    df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quoteAssetVolume","trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[["ts","open","high","low","close","volume","closeTime","quoteAssetVolume","trades"]]
    logging.info("Total bars fetched: %d", len(df))
    return df

# ---------------- resample helper (use 'min' and 'h' to avoid warnings) ----------------
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.set_index("ts").sort_index()
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out = out.dropna().reset_index()
    return out

# ---------------- indicators ----------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ---------------- semáforo signals ----------------
def generate_semaforo(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    df["ma10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma30"] = df["close"].rolling(window=30, min_periods=1).mean()
    df["rsi14"] = rsi(df["close"], 14)
    macd_line, sig_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = sig_line

    # Conditions
    cond_ma_long = df["ma10"] > df["ma30"]
    cond_rsi = df["rsi14"] > 50
    cond_macd = df["macd"] > df["macd_signal"]

    df["signal"] = "HOLD"
    df.loc[cond_ma_long & cond_rsi & cond_macd, "signal"] = "BUY"
    df.loc[~cond_ma_long & (~cond_rsi) & (df["macd"] < df["macd_signal"]), "signal"] = "SELL"

    return df

# ---------------- simple backtest for semáforo ----------------
def backtest_semaforo(signals_df: pd.DataFrame, cfg: Dict[str, Any]) -> (pd.DataFrame, Dict[str, Any]):
    balance = float(cfg.get("starting_balance", 100.0))
    fee = float(cfg.get("fee_rate", 0.0006))
    slippage = float(cfg.get("slippage", 0.0002))

    trades = []
    position = 0.0
    entry_price = None
    entry_idx = None

    for i, row in signals_df.iterrows():
        sig = row["signal"]
        price = row["close"]
        ts = row["ts"]
        # BUY: open position at next candle open
        if sig == "BUY" and position == 0:
            # enter at price * (1 + slippage)
            enter_price = price * (1 + slippage)
            qty = balance * 0.005 / enter_price  # fixed 0.5% of balance risked as starting simple sizing
            if qty * enter_price < 1.0:
                # avoid dust
                qty = max(qty, 0.000001)
            position = qty
            entry_price = enter_price
            entry_idx = i
            trades.append({
                "entry_time": ts, "entry_price": entry_price, "qty": qty,
                "exit_time": None, "exit_price": None, "pnl": None
            })
        # SELL: exit if in position or open a short (we don't short)
        if sig == "SELL" and position > 0:
            exit_price = price * (1 - slippage)
            last = trades[-1]
            last["exit_time"] = ts
            last["exit_price"] = exit_price
            gross = (exit_price - last["entry_price"]) * last["qty"]
            fees = (last["entry_price"] * last["qty"] + exit_price * last["qty"]) * fee
            net = gross - fees
            last["pnl"] = net
            balance += net
            position = 0.0
            entry_price = None
            entry_idx = None

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        win_rate = len(wins) / len(trades_df)
        total = trades_df["pnl"].sum()
        avg = trades_df["pnl"].mean()
    else:
        win_rate = 0.0; total = 0.0; avg = 0.0
    metrics = {"n_trades": len(trades_df), "win_rate": win_rate, "total_pnl": total, "avg_pnl": avg, "final_balance": balance}
    return trades_df, metrics

# ---------------- run pipeline ----------------
def run_pipeline():
    logging.info("Start pipeline - paginated 90 days fetch")
    # 1) fetch 90 days of 1m
    df_raw = fetch_klines_paginated(SYMBOL, INTERVAL, days=90, endpoints=ENDPOINTS, limit_per_call=1000)
    raw_path = os.path.join(DATA_FOLDER, "raw_1m.csv")
    df_raw.to_csv(raw_path, index=False)
    logging.info("Saved raw_1m.csv (%s rows)", len(df_raw))

    # 2) resample to 15m and 1h using pandas with modern aliases
    df_15m = resample_ohlcv(df_raw, "15min")
    df_1h  = resample_ohlcv(df_raw, "1h")
    df_15m.to_csv(os.path.join(DATA_FOLDER, "resampled_15m.csv"), index=False)
    df_1h.to_csv(os.path.join(DATA_FOLDER, "resampled_1h.csv"), index=False)
    logging.info("Saved resampled files: 15m(%d rows) 1h(%d rows)", len(df_15m), len(df_1h))

    # 3) compute semaforo on 15m
    df_signals = generate_semaforo(df_15m)
    df_signals.to_csv(os.path.join(OUT_FOLDER, "signals_15m.csv"), index=False)
    logging.info("Saved signals_15m.csv")

    # 4) backtest
    trades_df, metrics = backtest_semaforo(df_signals, CFG)
    trades_df.to_csv(os.path.join(OUT_FOLDER, "backtest.csv"), index=False)
    logging.info("Saved backtest.csv")
    logging.info("Metrics: %s", metrics)
    print("Run finished. Metrics:", metrics)

if __name__ == "__main__":
    run_pipeline()

