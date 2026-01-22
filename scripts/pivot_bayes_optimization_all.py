"""
pivot_bayes_optimization_all.py

Wersja "Human Readable":
- Zamiast Parallel Coordinates generuje zestaw Heatmap (Pairwise Analysis)
- Pozwala łatwo znaleźć stabilne strefy zysku
- Pełny Bayes (Beta-Binomial) + Open Execution
"""

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
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

SPREAD = 2.0
WARMUP = 50 

# Parametry Bayesa (Prior)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# SIATKA PARAMETRÓW
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
BUFFER_RANGE = [0.0010, 0.0020, 0.0030, 0.0040] 
BAYES_LONG_RANGE = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
BAYES_SHORT_RANGE = [0.50, 0.51, 0.52]

# ==========================================
# SILNIK
# ==========================================

def run_simulation(df: pd.DataFrame, sl_pct: float, buf_pct: float, b_long: float, b_short: float):
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    current_equity = 0.0
    
    history_s1 = deque(maxlen=200)
    history_r1 = deque(maxlen=200)
    
    bayes_denom_const = PRIOR_ALPHA + PRIOR_BETA
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        
        # Bayes Prob
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1)
        if n_r1 >= WARMUP:
            p_r1 = (PRIOR_ALPHA + k_r1) / (bayes_denom_const + n_r1)
        else:
            p_r1 = 0.5
            
        n_s1 = len(history_s1)
        k_s1 = sum(history_s1)
        if n_s1 >= WARMUP:
            p_s1 = (PRIOR_ALPHA + k_s1) / (bayes_denom_const + n_s1)
        else:
            p_s1 = 0.5
            
        # Sygnały
        curr_b = c_prev * buf_pct
        r1_val = (2*PP - l_prev)
        s1_val = (2*PP - h_prev)
        
        event_short = c_prev > (r1_val - curr_b)
        event_long = c_prev < (s1_val + curr_b)
        
        if event_short and event_long: event_short, event_long = False, False
        
        trade_dir = None
        if event_short and p_r1 > b_short: trade_dir = 'SHORT'
        elif event_long and p_s1 > b_long: trade_dir = 'LONG'
        
        # Exec (Open-Based)
        curr_sl = c_prev * sl_pct
        
        # Short
        raw_pnl_short = o_curr - c_curr
        is_sl_short = h_curr > (o_curr + curr_sl)
        real_pnl_short = (-curr_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        # Long
        raw_pnl_long = c_curr - o_curr
        is_sl_long = l_curr < (o_curr - curr_sl)
        real_pnl_long = (-curr_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
        
        if trade_dir == 'SHORT': current_equity += real_pnl_short
        elif trade_dir == 'LONG': current_equity += real_pnl_long
        
    return current_equity

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("Ładowanie danych...")
    sy, sm = int(START_DATE[:4]), int(START_DATE[5:7])
    ey, em = int(END_DATE[:4]), int(END_DATE[5:7])
    
    df = load_bars(SYMBOL, "1h", sy, sm, ey, em).loc[START_DATE:END_DATE]
    
    results = []
    total_iter = len(SL_RANGE)*len(BUFFER_RANGE)*len(BAYES_LONG_RANGE)*len(BAYES_SHORT_RANGE)
    
    print(f"Start Optymalizacji 4D ({total_iter} kombinacji)...")
    
    count = 0
    for sl in SL_RANGE:
        for buf in BUFFER_RANGE:
            for bl in BAYES_LONG_RANGE:
                for bs in BAYES_SHORT_RANGE:
                    pnl = run_simulation(df, sl, buf, bl, bs)
                    results.append({
                        'SL %': sl,
                        'Buffer %': buf,
                        'Bayes L': bl,
                        'Bayes S': bs,
                        'Profit': pnl
                    })
                    count += 1
                    if count % 20 == 0: print(f"{count}/{total_iter}...", end='\r')

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('Profit', ascending=False)
    
    res_df.to_csv('optimization_results.csv', index=False)
    print("\n\nTOP 5 WYNIKÓW:")
    print(res_df.head(5))
    
    # --- RYSOWANIE HEATMAP (CZYTELNE!) ---
    print("Generowanie zestawu Heatmap...")
    
    plt.figure(figsize=(18, 5))
    
    # 1. Heatmapa: SL vs Bayes Long (Agregacja średniej z Buffer i Bayes Short)
    plt.subplot(1, 3, 1)
    pivot_1 = res_df.pivot_table(index='Bayes L', columns='SL %', values='Profit', aggfunc='mean')
    # Odwracamy Y
    sns.heatmap(pivot_1.iloc[::-1], annot=True, fmt=".0f", cmap="RdYlGn")
    plt.title("Średni Profit: Bayes Long vs SL")
    
    # 2. Heatmapa: Buffer vs Bayes Long
    plt.subplot(1, 3, 2)
    pivot_2 = res_df.pivot_table(index='Bayes L', columns='Buffer %', values='Profit', aggfunc='mean')
    sns.heatmap(pivot_2.iloc[::-1], annot=True, fmt=".0f", cmap="RdYlGn")
    plt.title("Średni Profit: Bayes Long vs Buffer")

    # 3. Heatmapa: SL vs Buffer
    plt.subplot(1, 3, 3)
    pivot_3 = res_df.pivot_table(index='Buffer %', columns='SL %', values='Profit', aggfunc='mean')
    sns.heatmap(pivot_3.iloc[::-1], annot=True, fmt=".0f", cmap="RdYlGn")
    plt.title("Średni Profit: Buffer vs SL")
    
    plt.suptitle("Analiza Wrażliwości Parametrów (Agregacja Średnich)", fontsize=16)
    plt.tight_layout()
    
    plt.savefig("optimization_heatmaps.png")
    plt.show()
    
    print("Gotowe! Zapisano: optimization_heatmaps.png")
