"""
pivot_bayes_grid_dual.py

Zmodyfikowana wersja Dual Grid Search:
- Porównuje dwa różne OKNA PAMIĘCI BAYESA (zamiast buforów)
- Implementacja pełnego Bayesa (Beta-Binomial)
- Egzekucja transakcji na Open
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
B_PCT = 0.0020            # Stały bufor 0.2%
BAYES_MIN_EVENTS = 50     # Stały warmup
FIXED_SHORT_THRESHOLD = 0.52 # Stały próg dla Shortów

# Parametry Bayesa (Prior)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# === PARAMETRY DO BADANIA (DUAL) ===

# 1. Dwa warianty Okna Bayesa (Window Size)
# Sprawdzamy czy lepiej mieć krótką pamięć (100) czy długą (300)
WINDOW_SCENARIOS = [100, 300] 

# 2. Zakres Stop Loss
SL_RANGE = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]

# 3. Zakres Bayesa Long
BAYES_LONG_RANGE = [0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]

# ==========================================
# SILNIK (Z obsługą zmiennego okna)
# ==========================================

def run_simulation_fast(df: pd.DataFrame, sl_pct: float, bayes_long_threshold: float, window_size: int):
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    current_equity = 0.0
    
    # Używamy zmiennego rozmiaru okna przekazanego w argumencie
    history_s1 = deque(maxlen=window_size)
    history_r1 = deque(maxlen=window_size)
    
    bayes_denom_const = PRIOR_ALPHA + PRIOR_BETA
    
    for i in range(1, n):
        # DANE POPRZEDNIE
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        
        # DANE BIEŻĄCE
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        current_b = c_prev * B_PCT
        current_sl = c_prev * sl_pct
        
        # --- BAYES PROBABILITY ---
        # Short
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1)
        if n_r1 >= BAYES_MIN_EVENTS:
            p_r1 = (PRIOR_ALPHA + k_r1) / (bayes_denom_const + n_r1)
        else:
            p_r1 = 0.5 # Neutralne
            
        # Long
        n_s1 = len(history_s1)
        k_s1 = sum(history_s1)
        if n_s1 >= BAYES_MIN_EVENTS:
            p_s1 = (PRIOR_ALPHA + k_s1) / (bayes_denom_const + n_s1)
        else:
            p_s1 = 0.5
            
        # Sygnały
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        
        if event_short and event_long:
            event_short, event_long = False, False
            
        # Decyzja
        trade_dir = None
        
        if event_short and p_r1 > FIXED_SHORT_THRESHOLD:
            trade_dir = 'SHORT'
        elif event_long and p_s1 > bayes_long_threshold:
            trade_dir = 'LONG'
            
        # --- EGZEKUCJA (OPEN-BASED) ---
        
        # Short
        raw_pnl_short = o_curr - c_curr
        is_sl_short = h_curr > (o_curr + current_sl)
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        # Long
        raw_pnl_long = c_curr - o_curr
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        # Uczenie Bayesa
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
            
        print("Start Dual Grid Search (Window Analysis)...")
        
        # Przygotowanie wykresu (1 wiersz, 2 kolumny)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Pętla po dwóch wariantach okna (Window Size)
        for idx, win_size in enumerate(WINDOW_SCENARIOS):
            print(f"Obliczanie dla Window Size = {win_size}...")
            
            results = np.zeros((len(BAYES_LONG_RANGE), len(SL_RANGE)))
            
            for i, b_thresh in enumerate(BAYES_LONG_RANGE):
                for j, sl in enumerate(SL_RANGE):
                    # Przekazujemy win_size do symulacji
                    profit = run_simulation_fast(df, sl, b_thresh, win_size)
                    results[i, j] = profit
            
            # Rysowanie
            xticklabels = [f"{x*100:.1f}%" for x in SL_RANGE]
            yticklabels = [f"{y:.2f}" for y in BAYES_LONG_RANGE]
            
            sns.heatmap(results[::-1], annot=True, fmt=".0f", cmap="RdYlGn",
                       xticklabels=xticklabels, yticklabels=yticklabels[::-1],
                       ax=axes[idx])
            
            axes[idx].set_title(f"Bayes Window = {win_size} (Profit USD)")
            axes[idx].set_xlabel("Stop Loss (%)")
            axes[idx].set_ylabel("Bayes Threshold Long")
        
        plt.suptitle(f"Analiza Pamięci Bayesa: Szybka (100) vs Wolna (300)\nShortThreshold={FIXED_SHORT_THRESHOLD}, Buffer={B_PCT*100:.1f}%")
        plt.tight_layout()
        plt.savefig("heatmap_dual_window.png")
        plt.show()
        
        print("Gotowe! Zapisano jako heatmap_dual_window.png")
        
    except Exception as e:
        print(f"Błąd: {e}")
