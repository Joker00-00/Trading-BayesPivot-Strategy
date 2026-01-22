"""
pivot_bayes_validation_FINAL.py
-------------------------------
Skrypt walidacyjny zgodny z logiką MEAN-REVERSION z pliku pivot_bayes_strategy.py.
Służy do wygenerowania Tabeli 7 (Walidacja Cross-Asset).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque
import os

# ==========================================
# KONFIGURACJA
# ==========================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Parametry zwycięskie (z Grid Search dla BTC)
BEST_SL = 0.005           # 0.5%
BEST_BAYES_THRESH = 0.50  # Próg decyzyjny

# Stałe strategii (zgodne z pivot_bayes_strategy.py)
BAYES_WINDOW = 200
BAYES_MIN_EVENTS = 50
BPCT = 0.0040        # Bufor 0.4%
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# Koszty transakcyjne (Spread)
SPREAD_DICT = {
    "BTCUSDT": 2.0,
    "ETHUSDT": 0.20,
    "BNBUSDT": 0.05,
    "SOLUSDT": 0.03
}

# ==========================================
# LOADER DANYCH
# ==========================================
def load_data_simple(symbol):
    # Próba znalezienia danych w typowych lokalizacjach
    paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', symbol, '1h')),
        f"data/raw/{symbol}/1h",
        f"../data/raw/{symbol}/1h"
    ]
    
    root_dir = None
    for p in paths:
        if os.path.exists(p):
            root_dir = p
            break
            
    if not root_dir:
        print(f"[WARN] Nie znaleziono danych dla {symbol}")
        return pd.DataFrame()

    all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".csv")])
    dfs = []
    valid_years = ["2022", "2023", "2024", "2025"]
    
    print(f"Wczytywanie {symbol}...")
    for f in all_files:
        if any(y in f for y in valid_years):
            path = os.path.join(root_dir, f)
            try:
                df_temp = pd.read_csv(path, header=0)
                if df_temp.shape[1] >= 6:
                    df_temp = df_temp.iloc[:, :6]
                    df_temp.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    dfs.append(df_temp)
            except: pass
            
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Konwersja daty
    try:
        if df['timestamp'].iloc[0] > 2000000000:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    except: pass
    
    df = df.set_index('timestamp').sort_index()
    # Zakres dat
    df = df.loc[START_DATE:END_DATE]
    
    # Konwersja na float
    for col in ['open', 'high', 'low', 'close']: 
        df[col] = df[col].astype(float)
        
    return df

# ==========================================
# STRATEGIA (MEAN REVERSION)
# ==========================================
def run_strategy_mean_reversion(df, symbol, sl_pct, bayes_thresh):
    if df.empty: return 0,0,0,0

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    history_s1 = deque(maxlen=BAYES_WINDOW) # Dla Longów (S1)
    history_r1 = deque(maxlen=BAYES_WINDOW) # Dla Shortów (R1)
    
    equity = 0.0
    max_equity = -99999999.0
    max_drawdown = 0.0
    trades_count = 0
    wins_count = 0
    
    spread = SPREAD_DICT.get(symbol, 0.01)

    # Pętla po świecach
    for i in range(1, n):
        # i-1: Dane historyczne (Sygnał generowany na zamknięciu poprzedniej świecy)
        h_prev, l_prev, c_prev = highs[i-1], lows[i-1], closes[i-1]
        
        # i: Dane bieżące (Egzekucja na Open obecnej świecy)
        o_curr = opens[i]
        h_curr, l_curr, c_curr = highs[i], lows[i], closes[i]

        # Pivoty
        pp = (h_prev + l_prev + c_prev) / 3.0
        r1 = 2 * pp - l_prev
        s1 = 2 * pp - h_prev
        
        current_b = c_prev * BPCT
        current_sl = c_prev * sl_pct
        
        # LOGIKA MEAN-REVERSION (Zgodna z pivot_bayes_strategy.py)
        # Short: Cena jest wysoko (powyżej R1 - bufor) -> Gramy na powrót
        event_short = c_prev > (r1 - current_b)
        
        # Long: Cena jest nisko (poniżej S1 + bufor) -> Gramy na odbicie
        event_long  = c_prev < (s1 + current_b)
        
        # Wykluczenie sprzecznych
        if event_short and event_long:
            event_short, event_long = False, False

        # BAYES UPDATE (Obliczamy prawdopodobieństwo sukcesu)
        n_r1 = len(history_r1)
        k_r1 = sum(history_r1)
        if n_r1 >= BAYES_MIN_EVENTS:
            p_r1 = (PRIOR_ALPHA + k_r1) / (PRIOR_ALPHA + PRIOR_BETA + n_r1)
        else:
            p_r1 = 0.5 # Neutralne, póki nie mamy próbki
            
        n_s1 = len(history_s1)
        k_s1 = sum(history_s1)
        if n_s1 >= BAYES_MIN_EVENTS:
            p_s1 = (PRIOR_ALPHA + k_s1) / (PRIOR_ALPHA + PRIOR_BETA + n_s1)
        else:
            p_s1 = 0.5
            
        # DECYZJA
        trade_dir = None
        
        # Filtr Bayesa
        # Jeśli bayes_thresh > 0, filtrujemy. Jeśli 0.0 (RAW), bierzemy wszystko.
        if event_short:
             if bayes_thresh == 0.0 or p_r1 > bayes_thresh:
                 trade_dir = 'SHORT'
                 
        elif event_long:
             if bayes_thresh == 0.0 or p_s1 > bayes_thresh:
                 trade_dir = 'LONG'
            
        # OBLICZENIE PNL (Intraday: Entry=Open, Exit=Close)
        
        # Short PnL
        raw_pnl_short = o_curr - c_curr # Zysk, gdy cena spada
        is_sl_short = h_curr > (o_curr + current_sl) # Czy w trakcie świecy dotknęliśmy SL?
        real_pnl_short = -current_sl - spread if is_sl_short else raw_pnl_short - spread
        
        # Long PnL
        raw_pnl_long = c_curr - o_curr # Zysk, gdy cena rośnie
        is_sl_long = l_curr < (o_curr - current_sl)
        real_pnl_long = -current_sl - spread if is_sl_long else raw_pnl_long - spread
        
        # UCZENIE (FEEDBACK LOOP)
        # Bayes uczy się ZAWSZE, gdy wystąpił sygnał (nawet jeśli nie zagraliśmy)
        # To pozwala mu oceniać "co by było gdyby"
        if event_short:
            history_r1.append(1 if real_pnl_short > 0 else 0)
        if event_long:
            history_s1.append(1 if real_pnl_long > 0 else 0)
            
        # EGZEKUCJA (Tylko jeśli trade_dir != None)
        if trade_dir == 'SHORT':
            equity += real_pnl_short
            trades_count += 1
            if real_pnl_short > 0: wins_count += 1
        elif trade_dir == 'LONG':
            equity += real_pnl_long
            trades_count += 1
            if real_pnl_long > 0: wins_count += 1
            
        # Max DD tracking
        if equity > max_equity:
            max_equity = equity
        dd = max_equity - equity
        if dd > max_drawdown:
            max_drawdown = dd

    return equity, max_drawdown, trades_count, (wins_count/trades_count*100 if trades_count>0 else 0)

# ==========================================
# URUCHOMIENIE
# ==========================================
print(f"{'SYMBOL':<10} | {'MODE':<6} | {'PROFIT':>10} | {'DD':>10} | {'TRADES':>6} | {'WIN%':>5}")
print("-" * 65)

for symbol in SYMBOLS:
    df = load_data_simple(symbol)
    if df.empty:
        # Pusty wiersz, jeśli brak danych (dla testu)
        print(f"{symbol:<10} | NO DATA")
        continue
        
    # RAW (Bayes = 0.0 -> Wszystkie sygnały wchodzą)
    p_raw, dd_raw, t_raw, w_raw = run_strategy_mean_reversion(df, symbol, BEST_SL, 0.0)
    
    # BAYES (Bayes = 0.50 -> Filtrowanie poniżej 50%)
    p_bay, dd_bay, t_bay, w_bay = run_strategy_mean_reversion(df, symbol, BEST_SL, BEST_BAYES_THRESH)
    
    print(f"{symbol:<10} | RAW    | {p_raw:10.2f} | {dd_raw:10.2f} | {t_raw:6} | {w_raw:5.1f}%")
    print(f"{symbol:<10} | BAYES  | {p_bay:10.2f} | {dd_bay:10.2f} | {t_bay:6} | {w_bay:5.1f}%")
    print("-" * 65)
