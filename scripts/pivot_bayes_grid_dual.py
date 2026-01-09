from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
# Import loadera
try:
    from src.data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

SPREAD = 2.0
BAYES_MIN_EVENTS = 200
BAYES_WINDOW = 200
FIXED_SHORT_THRESHOLD = 0.51

# === PARAMETRY DO BADANIA ===

# 1. Dwa warianty Bufora (b)
BUFFER_SCENARIOS = [0.0010, 0.0030] # 0.1% i 0.3%

# 2. Zakres Stop Loss
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# 3. Zakres Bayesa Long
BAYES_LONG_RANGE = [0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

# ==========================================
# SILNIK
# ==========================================
def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float, b_pct: float):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    current_equity = 0.0
    history_s1 = deque(maxlen=BAYES_WINDOW)
    history_r1 = deque(maxlen=BAYES_WINDOW)
    
    min_events = BAYES_MIN_EVENTS
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        current_b = c_prev * b_pct # Używamy parametru z argumentu funkcji
        current_sl = c_prev * sl_pct
        
        p_r1 = np.mean(history_r1) if len(history_r1) >= min_events else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= min_events else 0.5
        
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        if event_short and event_long: event_short, event_long = False, False

        trade_dir = None
        if event_short and p_r1 > FIXED_SHORT_THRESHOLD: trade_dir = 'SHORT'
        elif event_long and p_s1 > bayes_long_threshold: trade_dir = 'LONG'

        # Exec
        raw_pnl_short = c_prev - c_curr
        is_sl_short = (h_curr - c_prev) > current_sl
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        raw_pnl_long = c_curr - c_prev
        is_sl_long = (c_prev - l_curr) > current_sl
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
        if trade_dir == 'SHORT': current_equity += real_pnl_short
        elif trade_dir == 'LONG': current_equity += real_pnl_long
            
    return current_equity

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Pobieranie danych {SYMBOL}...")
    try:
        sy, sm = int(START_DATE[:4]), int(START_DATE[5:7])
        ey, em = int(END_DATE[:4]), int(END_DATE[5:7])
        df = load_bars(SYMBOL, "1h", sy, sm, ey, em)
        df = df.loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych!")
            exit()
            
        print("Start Dual Grid Search...")
        
        # Przygotowanie wykresu (1 wiersz, 2 kolumny)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Pętla po dwóch wariantach bufora
        for idx, buffer_val in enumerate(BUFFER_SCENARIOS):
            print(f"Obliczanie dla Buffer = {buffer_val*100:.1f}%...")
            results = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
            
            for i, b_thresh in enumerate(BAYES_LONG_RANGE):
                for j, sl in enumerate(SL_RANGE):
                    profit = run_simulation_fast(df, sl, b_thresh, buffer_val)
                    results[i, j] = profit
            
            # Rysowanie na odpowiednim panelu (axes[idx])
            xticklabels = [f"{x*100:.1f}%" for x in SL_RANGE]
            yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
            
            # Odwracamy Y
            sns.heatmap(results[::-1], annot=True, fmt=".0f", cmap="RdYlGn", 
                        xticklabels=xticklabels, yticklabels=yticklabels[::-1], 
                        ax=axes[idx])
            
            axes[idx].set_title(f"Buffer = {buffer_val*100:.1f}% (Profit USD)")
            axes[idx].set_xlabel("Stop Loss (%)")
            axes[idx].set_ylabel("Bayes Threshold Long")

        plt.suptitle(f"Porównanie wpływu Bufora (b) na zyskowność\nShortThreshold={FIXED_SHORT_THRESHOLD}, Warmup={BAYES_MIN_EVENTS}")
        plt.tight_layout()
        plt.savefig("heatmap_dual.png")
        plt.show()
        
        print("Gotowe! Zapisano jako heatmap_dual.png")
        
    except Exception as e:
        print(f"Błąd: {e}")
