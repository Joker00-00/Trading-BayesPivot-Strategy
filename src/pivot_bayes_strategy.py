from __future__ import annotations
import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import modułów
try:
    from data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA (Zwycięska z Optymalizacji)
# ==========================================
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"

# Koszty
SPREAD = 2.0

# Parametry Strategii (Finalne)
SL_PCT = 0.0150   # SL 2.0%
B_PCT = 0.0020    # Buffer 0.4%

# Parametry Bayesa
BAYES_WINDOW = 300
BAYES_MIN_EVENTS = 200
BAYES_THRESHOLD_LONG = 0.52
BAYES_THRESHOLD_SHORT = 0.52

OUTPUT_DIR = "trades"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def parse_date_range(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    return start.year, start.month, end.year, end.month

def calculate_max_drawdown(equity_curve):
    peak = -999999999.0
    max_dd = 0.0
    for value in equity_curve:
        if value > peak: peak = value
        dd = value - peak
        if dd < max_dd: max_dd = dd
    return max_dd

def calculate_calmar_ratio(total_profit, max_dd):
    if max_dd == 0: return 0.0
    return total_profit / abs(max_dd)

# ==========================================
# 1. ANALIZA GLOBALNA (A Priori)
# ==========================================
def calculate_global_stats(df: pd.DataFrame):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    kl, ks = 0, 0
    kr, klr = 0, 0
    ks_s1, kls = 0, 0

    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        c_curr = closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2*PP - l_prev
        S1 = 2*PP - h_prev
        
        current_b = c_prev * B_PCT
        
        if c_curr > c_prev: kl += 1
        elif c_curr < c_prev: ks += 1
            
        if c_prev > (R1 - current_b):
            kr += 1
            if c_curr < c_prev: klr += 1
                
        if c_prev < (S1 + current_b):
            ks_s1 += 1
            if c_curr > c_prev: kls += 1

    prob_up = kl / (kl + ks) if (kl+ks) > 0 else 0
    plr_r1 = klr / kr if kr > 0 else 0
    plr_s1 = kls / ks_s1 if ks_s1 > 0 else 0

    print("\n=== ANALIZA GLOBALNA (A PRIORI) ===")
    print(f"Zakres: {START_DATE} do {END_DATE}")
    print(f"Bias rynku (Bullish): {prob_up:.2%}")
    print(f"[R1 SHORT] Skuteczność: {plr_r1:.2%} (na {kr} okazji)")
    print(f"[S1 LONG]  Skuteczność: {plr_s1:.2%} (na {ks_s1} okazji)")
    print("===================================\n")

# ==========================================
# 2. STRATEGIA (POPRAWIONA: PnL od Open)
# ==========================================
def run_strategy(df: pd.DataFrame):
    # Pobieramy również ceny OPEN dla precyzji
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    n = len(df)
    
    equity_curve = [0.0]
    trades_log = []
    
    history_r1 = deque(maxlen=BAYES_WINDOW)
    history_s1 = deque(maxlen=BAYES_WINDOW)
    
    current_equity = 0.0
    
    print(f"Start symulacji (SL={SL_PCT*100}%, Bayes L>{BAYES_THRESHOLD_LONG})...")

    for i in range(1, n):
        # Dane poprzednie (do sygnału)
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        
        # Dane obecne (do egzekucji)
        o_curr = opens[i]  # Cena wejścia (Open świecy)
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2 * PP - l_prev
        S1 = 2 * PP - h_prev
        
        current_b = c_prev * B_PCT
        current_sl = c_prev * SL_PCT
        
        p_r1 = np.mean(history_r1) if len(history_r1) >= BAYES_MIN_EVENTS else 0.5
        p_s1 = np.mean(history_s1) if len(history_s1) >= BAYES_MIN_EVENTS else 0.5
        
        event_short = c_prev > (R1 - current_b)
        event_long = c_prev < (S1 + current_b)
        
        if event_short and event_long: event_short, event_long = False, False

        trade_dir = None
        if event_short and p_r1 > BAYES_THRESHOLD_SHORT: trade_dir = 'SHORT'
        elif event_long and p_s1 > BAYES_THRESHOLD_LONG: trade_dir = 'LONG'

        # --- POPRAWIONE OBLICZENIA PNL (od Open) ---
        
        # SHORT EXECUTION
        # Zysk brutto: Open - Close (bo sprzedajemy na Open, odkupujemy na Close)
        raw_pnl_short = o_curr - c_curr
        # Stop Loss: czy High świecy przebił (Open + SL)
        is_sl_short = h_curr > (o_curr + current_sl)
        # Wynik: albo strata SL, albo wynik z zamknięcia
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        # LONG EXECUTION
        # Zysk brutto: Close - Open
        raw_pnl_long = c_curr - o_curr
        # Stop Loss: czy Low świecy przebił (Open - SL)
        is_sl_long = l_curr < (o_curr - current_sl)
        # Wynik
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
        # --- KONIEC POPRAWEK ---

        if event_short: history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long: history_s1.append(1 if real_pnl_long > 0 else 0)
            
        if trade_dir == 'SHORT':
            current_equity += real_pnl_short
            trades_log.append({'time': times[i], 'type': 'SHORT', 'price': o_curr, 'pnl': real_pnl_short, 'equity': current_equity})
        elif trade_dir == 'LONG':
            current_equity += real_pnl_long
            trades_log.append({'time': times[i], 'type': 'LONG', 'price': o_curr, 'pnl': real_pnl_long, 'equity': current_equity})
            
        equity_curve.append(current_equity)

    return pd.DataFrame(trades_log), equity_curve

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        sy, sm, ey, em = parse_date_range(START_DATE, END_DATE)
        print(f"Pobieranie: {SYMBOL}...")
        df = load_bars(SYMBOL, INTERVAL, sy, sm, ey, em)
        df = df.loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych.")
            exit()
            
        calculate_global_stats(df)
        trades_df, equity = run_strategy(df)
        
        filename = f"{OUTPUT_DIR}/trades_final_{SYMBOL}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"Zapisano logi: {filename}")
        
        if not trades_df.empty:
            total_profit = equity[-1]
            max_dd = calculate_max_drawdown(equity)
            calmar = calculate_calmar_ratio(total_profit, max_dd)
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (wins / len(trades_df)) * 100
            
            print("\n=== WYNIKI KOŃCOWE (POPRAWIONE) ===")
            print(f"Zysk netto:         {total_profit:.2f} USD")
            print(f"Max Drawdown:       {max_dd:.2f} USD")
            print(f"Win Rate:           {win_rate:.2f}%")
            print("-" * 30)
            print(f"CALMAR RATIO:       {calmar:.2f}")
            print("======================")
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity, label='Equity Curve', color='green')
            plt.title(f'Bayes Strategy Final - {SYMBOL} (Calmar: {calmar:.2f})')
            plt.xlabel('Bars')
            plt.ylabel('Profit (USD)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            
    except Exception as e:
        print(f"Błąd: {e}")
