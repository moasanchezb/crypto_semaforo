#!/usr/bin/env python3
"""
Grid Search para optimizar semáforo ponderado
Busca set de parámetros más rentable
"""

import pandas as pd
import numpy as np
import itertools
import logging

# ----- CONFIG -----
INPUT_FILE = "data/resampled_15m.csv"
OUTPUT_FILE = "gridsearch_results.csv"
POSITION_SIZE = 1.0

# Rango de parámetros a probar
EMA_SHORT_LIST = [50, 75, 100]
EMA_LONG_LIST = [200, 300, 400]
RSI_OVERSOLD_LIST = [30, 35, 40]
RSI_OVERBOUGHT_LIST = [60, 65, 70]
ATR_MULTIPLIER_LIST = [1.5, 2.0, 2.5]
WEIGHT_EMA_LIST = [0.3,0.4,0.5]
WEIGHT_RSI_LIST = [0.3,0.4,0.5]
WEIGHT_ATR_LIST = [0.1,0.2,0.3]

logging.basicConfig(level=logging.INFO)

# ----- FUNCIONES -----
def compute_indicators(df, ema_short, ema_long, rsi_period=14, atr_period=14):
    df = df.copy()
    df["EMA50"] = df["close"].ewm(span=ema_short, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=ema_long, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-Cprev"] = abs(df["high"] - df["close"].shift(1))
    df["L-Cprev"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L","H-Cprev","L-Cprev"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(atr_period).mean()
    df.drop(columns=["H-L","H-Cprev","L-Cprev","TR"], inplace=True)
    
    # Pendiente EMA50
    df["EMA50_slope"] = df["EMA50"].diff()
    
    return df

def generate_signals(df, rsi_oversold, rsi_overbought, atr_multiplier, weight_ema, weight_rsi, weight_atr):
    df = df.copy()
    semaforo = []
    score_list = []
    
    for idx, row in df.iterrows():
        score = 0.0
        
        # EMA
        ema_cond = row["EMA50"] > row["EMA200"]
        ema_slope = row["EMA50_slope"] if not pd.isna(row["EMA50_slope"]) else 0
        if ema_cond and ema_slope > 0:
            score += weight_ema
        elif not ema_cond and ema_slope < 0:
            score -= weight_ema
        
        # RSI
        if row["RSI"] < rsi_oversold:
            score += weight_rsi
        elif row["RSI"] > rsi_overbought:
            score -= weight_rsi
        
        # ATR
        if row["ATR"] < atr_multiplier:
            score += weight_atr
        else:
            score -= weight_atr
        
        score_list.append(score)
        
        # Semáforo
        if score >= 0.6:
            semaforo.append("BUY")
        elif score <= -0.6:
            semaforo.append("SELL")
        else:
            semaforo.append("HOLD")
    
    df["Signal"] = semaforo
    df["Score"] = score_list
    return df

def backtest(df):
    df = df.copy()
    df["Position"] = df["Signal"].shift(1).map({"BUY":1,"SELL":-1,"HOLD":0})
    df["Returns"] = df["close"].pct_change() * df["Position"] * POSITION_SIZE
    df["Equity"] = 100 + df["Returns"].cumsum()
    n_trades = df["Position"].abs().sum()
    win_rate = (df["Returns"] > 0).sum() / n_trades if n_trades>0 else 0
    total_pnl = df["Returns"].sum()
    avg_pnl = df["Returns"].mean() if n_trades>0 else 0
    final_balance = df["Equity"].iloc[-1]
    metrics = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "final_balance": final_balance
    }
    return metrics

# ----- GRID SEARCH -----
def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["ts"])
    results = []

    combos = list(itertools.product(
        EMA_SHORT_LIST,
        EMA_LONG_LIST,
        RSI_OVERSOLD_LIST,
        RSI_OVERBOUGHT_LIST,
        ATR_MULTIPLIER_LIST,
        WEIGHT_EMA_LIST,
        WEIGHT_RSI_LIST,
        WEIGHT_ATR_LIST
    ))

    logging.info(f"INFO: Total combinaciones: {len(combos)}")

    for i, (ema_s, ema_l, rsi_os, rsi_ob, atr_mult, w_ema, w_rsi, w_atr) in enumerate(combos,1):
        df_ind = compute_indicators(df, ema_s, ema_l)
        df_sig = generate_signals(df_ind, rsi_os, rsi_ob, atr_mult, w_ema, w_rsi, w_atr)
        metrics = backtest(df_sig)
        results.append({
            "EMA_SHORT": ema_s,
            "EMA_LONG": ema_l,
            "RSI_OVERSOLD": rsi_os,
            "RSI_OVERBOUGHT": rsi_ob,
            "ATR_MULT": atr_mult,
            "WEIGHT_EMA": w_ema,
            "WEIGHT_RSI": w_rsi,
            "WEIGHT_ATR": w_atr,
            **metrics
        })
        if i % 10 == 0:
            logging.info(f"INFO: Iteración {i}/{len(combos)}")

    df_res = pd.DataFrame(results)
    df_res.sort_values("total_pnl", ascending=False, inplace=True)
    df_res.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"INFO: Resultados guardados en {OUTPUT_FILE}")
    logging.info(f"INFO: Mejor combinación: {df_res.iloc[0].to_dict()}")

if __name__ == "__main__":
    main()

