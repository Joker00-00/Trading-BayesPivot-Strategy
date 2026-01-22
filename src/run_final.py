import sys
import pandas as pd
import numpy as np
from collections import deque

# --- IMPORT DANYCH ---
try:
    from data_loader import load_bars
except ImportError:
    print("Brak data_loader.py! Uruchom w folderze projektu.")
    sys.exit()

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# --- SILNIK 1: EXTENDED (Z pliku pivot_bayes_grid_search_extended.py) ---
# Służy do precyzyjnego dostrajania BTC
def run_extended_sim(df, sl_pct, bayes_thresh):
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    current_equity = 0.0
    history_s1 = deque(maxlen=200) # BAYES_WINDOW
    history_r1 = deque(maxlen=200)
    bayes_denom = 2.0 # Alpha+Beta (1+1)
    
    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr, h_curr, l_curr, c_curr = opens[i], highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1, S1 = 2*PP - l_prev, 2*PP - h_prev
        
        # Bayes Prob Update
        n_r1, k_r1 = len(history_r1), sum(history_r1)
        p_r1 = (1.0 + k_r1)/(bayes_denom + n_r1) if n_r1 >= 50 else 0.5
        
        n_s1, k_s1 = len(history_s1), sum(history_s1)
        p_s1 = (1.0 + k_s1)/(bayes_denom + n_s1) if n_s1 >= 50 else 0.5
        
        # Sygnały
        b = c_prev * 0.0035 # BUFFER z pliku extended
        ev_short = c_prev > (R1 - b)
        ev_long = c_prev < (S1 + b)
        if ev_short and ev_long: ev_short, ev_long = False, False
        
        trade = None
        if ev_short and p_r1 > 0.52: trade = 'SHORT' # Fixed Short Threshold
        elif ev_long and p_s1 > bayes_thresh: trade = 'LONG'
        
        # Exec
        sl_val = c_prev * sl_pct
        pnl = 0
        if trade == 'SHORT':
            raw = o_curr - c_curr
            is_sl = h_curr > (o_curr + sl_val)
            pnl = (-sl_val - 2.0) if is_sl else (raw - 2.0)
            if ev_short: history_r1.append(1 if pnl > 0 else 0)
        elif trade == 'LONG':
            raw = c_curr - o_curr
            is_sl = l_curr < (o_curr - sl_val)
            pnl = (-sl_val - 2.0) if is_sl else (raw - 2.0)
            if ev_long: history_s1.append(1 if pnl > 0 else 0)
            
        current_equity += pnl
        
    return current_equity

# --- SILNIK 2: DUAL/4D (Z pliku pivot_bayes_optimization_all.py) ---
# Służy do ratowania ALTCOINÓW (osobne progi)
def run_dual_sim(df, sl_pct, b_long, b_short):
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    equity = [0.0]
    history_s1 = deque(maxlen=200)
    history_r1 = deque(maxlen=200)
    
    current_eq = 0.0
    
    for i in range(1, n):
        # ... (Logika identyczna jak wyżej, ale z dwoma progami)
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr, h_curr, l_curr, c_curr = opens[i], highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        
        n_r1, k_r1 = len(history_r1), sum(history_r1)
        p_r1 = (1.0 + k_r1)/(2.0 + n_r1) if n_r1 >= 50 else 0.5
        
        n_s1, k_s1 = len(history_s1), sum(history_s1)
        p_s1 = (1.0 + k_s1)/(2.0 + n_s1) if n_s1 >= 50 else 0.5
        
        b = c_prev * 0.002 # Standardowy bufor
        ev_short = c_prev > ((2*PP - l_prev) - b)
        ev_long = c_prev < ((2*PP - h_prev) + b)
        if ev_short and ev_long: ev_short, ev_long = False, False
        
        trade = None
        if ev_short and p_r1 > b_short: trade = 'SHORT'
        elif ev_long and p_s1 > b_long: trade = 'LONG'
        
        sl_val = c_prev * sl_pct
        pnl = 0
        spread = 0.2 if df['close'].mean() < 5000 else 2.0 # Auto spread adjustment
        if df['close'].mean() < 300: spread = 0.05 # BNB/SOL
        
        if trade == 'SHORT':
            is_sl = h_curr > (o_curr + sl_val)
            pnl = (-sl_val - spread) if is_sl else (o_curr - c_curr - spread)
            if ev_short: history_r1.append(1 if pnl > 0 else 0)
        elif trade == 'LONG':
            is_sl = l_curr < (o_curr - sl_val)
            pnl = (-sl_val - spread) if is_sl else (c_curr - o_curr - spread)
            if ev_long: history_s1.append(1 if pnl > 0 else 0)
            
        current_eq += pnl
        equity.append(current_eq)
        
    # Calmar Calc
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = peak - eq_arr
    max_dd = np.max(dd)
    calmar = current_eq / max_dd if max_dd > 0 else 0
    return current_eq, max_dd, calmar

# --- MAIN ---
print("=== GENEROWANIE DANYCH DO FINALNEJ CZĘŚCI ROZDZIAŁU 5 ===\n")

# 1. EXTENDED BTC
print("1. BTC EXTENDED (Fine Tuning)...")
df_btc = load_bars("BTCUSDT", "1h", 2022, 1, 2025, 12).loc[START_DATE:END_DATE]
# Sprawdzamy kilka 'ciasnych' ustawień z pliku extended
best_btc = -999
best_cfg = ""
for sl in [0.005, 0.008, 0.010]:
    for bl in [0.52, 0.54, 0.55]:
        p = run_extended_sim(df_btc, sl, bl)
        if p > best_btc:
            best_btc = p
            best_cfg = f"SL={sl*100}% BayesLong>{bl}"
print(f"   -> Najlepszy wynik Extended: ${best_btc:.2f} ({best_cfg})")

# 2. DUAL ETH/SOL
print("\n2. DUAL OPTIMIZATION (Asymetria)...")
for sym in ["ETHUSDT", "SOLUSDT"]:
    df_coin = load_bars(sym, "1h", 2022, 1, 2025, 12).loc[START_DATE:END_DATE]
    # Szukamy konfiguracji: Long > 0.50 (agresywnie), Short > 0.55 (ostrożnie)
    p, dd, c = run_dual_sim(df_coin, 0.015, 0.50, 0.55) # Przykładowa asymetria
    print(f"   -> {sym} (Dual: L>0.50, S>0.55): Profit=${p:.2f}, MaxDD=${dd:.2f}, Calmar={c:.2f}")

print("\nGotowe. Skopiuj te liczby.")
