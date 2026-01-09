from __future__ import annotations
import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import loadera
try:
    from data_loader import load_bars
except ImportError:
    print("Błąd: Brak pliku data_loader.py.")
    exit()

# ==========================================
# KONFIGURACJA
# ==========================================
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
SPREAD = 2.0

# Parametry Finalne
SL_PCT = 0.0150   
B_PCT = 0.0020    
BAYES_WINDOW = 100
BAYES_MIN_EVENTS = 100
BAYES_THRESHOLD_LONG = 0.56
BAYES_THRESHOLD_SHORT = 0.51

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
        R1, S1 = 2*PP - l_prev, 2*PP - h_prev
        
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
# STRATEGIA (POPRAWIONA - PnL od Open)
# ==========================================
def run_strategy(df: pd.DataFrame):
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
    
    print(f"Start symulacji...")

    for i in range(1, n):
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]
        
        PP = (h_prev + l_prev + c_prev) / 3.0
        R1 = 2*PP - l_prev
        S1 = 2*PP - h_prev
        
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

        # Obliczenia PnL (Open-based)
        raw_pnl_short = o_curr - c_curr
        is_sl_short = h_curr > (o_curr + current_sl)
        real_pnl_short = (-current_sl - SPREAD) if is_sl_short else (raw_pnl_short - SPREAD)
        
        raw_pnl_long = c_curr - o_curr
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = (-current_sl - SPREAD) if is_sl_long else (raw_pnl_long - SPREAD)
        
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
        df = load_bars(SYMBOL, INTERVAL, sy, sm, ey, em).loc[START_DATE:END_DATE]
        
        if df.empty:
            print("Brak danych.")
            exit()
            
        calculate_global_stats(df)
        trades_df, equity = run_strategy(df)
        
        if not trades_df.empty:
            total_profit = equity[-1]
            max_dd = calculate_max_drawdown(equity)
            calmar = calculate_calmar_ratio(total_profit, max_dd)
            
            print(f"\nWynik: {total_profit:.2f} USD, Calmar: {calmar:.2f}")

            # ---------------------------------------------
            # WYKRES 1: CENA (PRICE) - Liniowy (czytelniejszy)
            # ---------------------------------------------
            plt.figure(figsize=(14, 6))
            plt.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
            plt.title(f"{SYMBOL} Price Action (2024)")
            plt.ylabel("Price ($)")
            plt.xlabel("Date")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("price_chart.png") # Zapis do pliku
            print("Zapisano: price_chart.png")
            
            # ---------------------------------------------
            # WYKRES 2: EQUITY (KAPITAŁ)
            # ---------------------------------------------
            # Wyrównanie danych
            equity_plot = equity[1:] 
            min_len = min(len(df)-1, len(equity_plot))
            equity_series = pd.Series(equity_plot[:min_len], index=df.index[1:min_len+1])

            plt.figure(figsize=(14, 6))
            plt.plot(equity_series.index, equity_series.values, label='Equity Curve', color='green', linewidth=1.5)
            plt.title(f"Strategy Equity Curve (Profit: ${total_profit:.0f}, Calmar: {calmar:.2f})")
            plt.ylabel("Total Profit ($)")
            plt.xlabel("Date")
            
            # Dodanie ramki z wynikami
            info_text = (f"Profit: ${total_profit:.0f}\n"
                         f"MaxDD: ${max_dd:.0f}\n"
                         f"Calmar: {calmar:.2f}")
            plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                         verticalalignment='top')

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("equity_chart.png") # Zapis do pliku
            print("Zapisano: equity_chart.png")
            
            plt.show() # Pokaż oba

    except Exception as e:
        print(f"Błąd: {e}")
