#!/usr/bin/env python3
"""
Semáforo optimizado de trading crypto
Usa resampled_15m.csv para generar señales BUY/HOLD/SELL
y calcula backtest automático.
"""

import pandas as pd
import numpy as np

# ----- CONFIGURACIÓN -----
INPUT_FILE = "data/resampled_15m.csv"
SIGNALS_FILE = "signals_15m_opt.csv"
BACKTEST_FILE = "backtest_opt.csv"

# Parámetros del semáforo
EMA_SHORT = 50
EMA_LONG = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5

# Tamaño de posición (1 unidad por trade)
POSITION_SIZE = 1.0

# ----- FUNCIONES -----
def compute_indicators(df):
    """Calcula EMA, RSI y ATR"""
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
    df["TR"] = df[["H-L", "H-Cprev", "L-Cprev"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()
    
    df.drop(columns=["H-L","H-Cprev","L-Cprev","TR"], inplace=True)
    return df

def generate_signals(df):
    """Genera semáforo tipo BUY/HOLD/SELL"""
    signals = []
    for idx, row in df.iterrows():
        ema_cond = row["EMA50"] > row["EMA200"]
        rsi_cond_buy = row["RSI"] < RSI_OVERSOLD
        rsi_cond_sell = row["RSI"] > RSI_OVERBOUGHT
        atr_cond = row["ATR"] > 0  # opcional para filtrar ATR extremo

        # Semáforo combinado
        if ema_cond and rsi_cond_buy and atr_cond:
            signals.append("BUY")
        elif not ema_cond and rsi_cond_sell and atr_cond:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    df["Signal"] = signals
    return df

def backtest(df):
    """Calcula PnL simple por trade"""
    df["Position"] = df["Signal"].shift(1).map({"BUY": 1, "SELL": -1, "HOLD": 0})
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
    df = generate_signals(df)
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

