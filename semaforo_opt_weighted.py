#!/usr/bin/env python3
"""
Semáforo optimizado con pesos y filtros de EMA/ATR
Genera señales BUY/HOLD/SELL (verde/amarillo/rojo)
Backtest incluido.
"""

import pandas as pd
import numpy as np

# ----- CONFIGURACIÓN -----
INPUT_FILE = "data/resampled_15m.csv"  # ruta de tu CSV
SIGNALS_FILE = "signals_15m_weighted.csv"
BACKTEST_FILE = "backtest_weighted.csv"

# Parámetros del semáforo
EMA_SHORT = 50
EMA_LONG = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
ATR_PERIOD = 14
ATR_MULTIPLIER_MAX = 2.0   # filtra volatilidad extrema
POSITION_SIZE = 1.0

# Pesos de los indicadores
WEIGHT_EMA = 0.4
WEIGHT_RSI = 0.4
WEIGHT_ATR = 0.2

# ----- FUNCIONES -----
def compute_indicators(df):
    df["EMA50"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-Cprev"] = abs(df["high"] - df["close"].shift(1))
    df["L-Cprev"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L","H-Cprev","L-Cprev"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()
    
    df.drop(columns=["H-L","H-Cprev","L-Cprev","TR"], inplace=True)

    # Pendiente EMA50 para semáforo
    df["EMA50_slope"] = df["EMA50"].diff()
    
    return df

def generate_weighted_signals(df):
    semaforo = []
    score_list = []

    for idx, row in df.iterrows():
        score = 0.0

        # EMA: dirección
        ema_cond = row["EMA50"] > row["EMA200"]
        ema_slope = row["EMA50_slope"] if not pd.isna(row["EMA50_slope"]) else 0
        if ema_cond and ema_slope > 0:
            score += WEIGHT_EMA
        elif not ema_cond and ema_slope < 0:
            score -= WEIGHT_EMA

        # RSI
        if row["RSI"] < RSI_OVERSOLD:
            score += WEIGHT_RSI
        elif row["RSI"] > RSI_OVERBOUGHT:
            score -= WEIGHT_RSI

        # ATR: filtra volatilidad extrema
        if row["ATR"] < ATR_MULTIPLIER_MAX:
            score += WEIGHT_ATR
        else:
            score -= WEIGHT_ATR

        score_list.append(score)

        # Semáforo tipo verde/amarillo/rojo
        if score >= 0.6:
            semaforo.append("BUY")      # Verde
        elif score <= -0.6:
            semaforo.append("SELL")     # Rojo
        else:
            semaforo.append("HOLD")     # Amarillo

    df["Signal"] = semaforo
    df["Score"] = score_list
    return df

def backtest(df):
    df["Position"] = df["Signal"].shift(1).map({"BUY":1,"SELL":-1,"HOLD":0})
    df["Returns"] = df["close"].pct_change() * df["Position"] * POSITION_SIZE
    df["Equity"] = 100 + df["Returns"].cumsum()
    n_trades = df["Position"].abs().sum()
    win_rate = (df["Returns"] > 0).sum() / n_trades if n_trades > 0 else 0
    total_pnl = df["Returns"].sum()
    avg_pnl = df["Returns"].mean() if n_trades > 0 else 0
    final_balance = df["Equity"].iloc[-1]
    metrics = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "final_balance": final_balance
    }
    return metrics, df

# ----- PIPELINE -----
def main():
    print("INFO: Cargando datos...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["ts"])
    df = compute_indicators(df)
    df = generate_weighted_signals(df)
    metrics, df_bt = backtest(df)

    print(f"INFO: Guardando señales en {SIGNALS_FILE}")
    df.to_csv(SIGNALS_FILE, index=False)
    print(f"INFO: Guardando backtest en {BACKTEST_FILE}")
    df_bt.to_csv(BACKTEST_FILE, index=False)

    print("INFO: MÉTRICAS")
    for k,v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

