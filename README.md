# Magisterka – BayesPivot (Pivot Points + filtr Bayes)

Repozytorium zawiera kod do pobierania i ładowania danych OHLCV oraz testowania strategii **Pivot Points** z opcjonalnym filtrem **Bayesowskim** (Beta-Binomial Conjugate Prior). Dodatkowo dostępne są skrypty do **grid search / optymalizacji** parametrów.

## Struktura projektu

src/ # rdzeń: downloader, loader, główna strategia
scripts/ # uruchamialne grid search / optymalizacje
data/ # dane rynkowe (lokalnie, gitignored)
trades/ # wyniki/logi transakcji (lokalnie, gitignored)

> `data/`, `trades/` oraz `__pycache__/` są ignorowane przez `.gitignore`.

## Wymagania

* Python **3.11+**
* Pakiety: `numpy`, `pandas`, `matplotlib`, `seaborn`, `requests`

## Uruchamianie

### Główna Strategia (Backtest)

Uruchamia backtest na pełnych danych z wizualizacją Equity Curve.

```bash
python src/pivot_bayes_strategy.py
```
Grid Search / Optymalizacja
Skrypty znajdują się w folderze scripts/. Każdy z nich ma na górze sekcję KONFIGURACJA, gdzie można łatwo zmienić interwał (1h, 15m) i zakresy dat.

1. Standardowy Grid Search (Wstępny)
Szybki przegląd szerokich zakresów parametrów.

```bash
python scripts/pivot_bayes_grid_search.py
```
2. Extended Grid Search ("Snajper")
Precyzyjne badanie najbardziej zyskownych obszarów (Fine Tuning), zidentyfikowanych we wstępnych testach.

```bash
python scripts/pivot_bayes_grid_search_extended.py
```
3. Dual Grid Search (Window Analysis)
Porównuje wpływ długości okna pamięci Bayesa (np. 100 vs 300 świec) na wyniki.

```bash
python scripts/pivot_bayes_grid_dual.py
```
4. Multi-Parameter Optimization (Heatmaps)
Generuje zestaw map ciepła (Pairwise Analysis) dla 4 parametrów jednocześnie (SL, Buffer, Bayes Long, Bayes Short), pozwalając znaleźć globalne optimum.

```bash
python scripts/pivot_bayes_optimization_all.py
```
Dane
Dane świec (Klines) są pobierane automatycznie (przez data_downloader.py wbudowany w loadera) lub przechowywane lokalnie w data/raw/....
Źródło danych: Binance Vision.