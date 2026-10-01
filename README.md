# Copper Brain v2 - Copper Direction Model (research)

<!-- one-tap-install -->
[![Download ZIP](https://img.shields.io/badge/Download-ZIP-2ea44f?style=for-the-badge&logo=github)](https://github.com/pensebastian072/Copper-DXY-ML/archive/refs/heads/main.zip)

**Run it on your computer in 3 steps:** 1) [download the ZIP](https://github.com/pensebastian072/Copper-DXY-ML/archive/refs/heads/main.zip) · 2) unzip it · 3) double-click **`install.bat`** (Windows) or run **`./install.sh`** (macOS/Linux).
The Streamlit app opens in your browser at `http://127.0.0.1:8501` - it runs only on your machine. Next time use `start.bat` / `./start.sh`.
The first run downloads market data from Yahoo Finance and trains the model (a few minutes).
<!-- one-tap-install -->

An XGBoost classifier for the **21-day direction** (up/down) of COMEX copper futures, evaluated with walk-forward validation. The [Results](#results) section shows what the evidence says - including that, so far, it does **not** beat simply predicting "up".

## Overview

Copper Brain v2 models **21-day ahead price direction** for COMEX copper futures using:
- Free daily data from yfinance
- Technical features including DXY and gold relationships
- Walk-forward rolling validation (a 1,260-calendar-day (~3.45-year) training window and a 21-calendar-day (~15 trading days) test step)
- An XGBoost binary classifier

## Project Structure

```
copper_brain_v2/
├── app.py                 # Streamlit dashboard
├── copper_brain_v2.py     # Core model training + backtest
├── features.py            # Feature engineering functions
├── model_utils.py         # Training, walk-forward, performance metrics
├── data/                  # Cached data files
├── outputs/               # Backtest results, model artifacts
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model & Run Backtest

```bash
python copper_brain_v2.py
```

This will:
- Download copper, DXY, and gold data from yfinance
- Engineer all technical features
- Run walk-forward validation (1,260-calendar-day rolling training window)
- Save results to `outputs/backtest_results.csv`
- Save trained model to `outputs/final_model.pkl`

### 2. Launch Dashboard

```bash
streamlit run app.py
```

Or if streamlit is not in PATH:

```bash
python -m streamlit run app.py
```

## Features

### Data Sources (Free Only)
- **Copper futures**: HG=F (COMEX)
- **US Dollar Index**: DX-Y.NYB
- **Gold futures**: GC=F

### Feature Groups
1. **Momentum**: 1d, 2d, 3d, 5d, 10d returns, rolling max/min
2. **Moving Averages**: 5d, 10d, 20d MAs and price ratios
3. **Volatility**: 10d and 21d realized volatility
4. **Volume**: Log volume, z-score, OBV
5. **Candle Structure**: Body, wicks, bullish/bearish
6. **Alpha Features**: DXY correlation, copper/gold ratio

### Model Architecture
- XGBoost Binary Classifier
- 500 trees, max depth 6, learning rate 0.03
- Walk-forward validation: 1,260-calendar-day (~3.45-year) training window, 21-calendar-day test step.
  (The code computes 5 x 252 trading days but applies the result as calendar days, so the window is
  ~3.45 years, not the 5 years it was meant to be.)
- Target: direction of the close 21 trading days ahead
- Probability threshold tuned on each training window (range 0.35-0.65)

## Results

From one full walk-forward run (trained 2026-09-28 on yfinance data; predictions dated
2004-03-15 to 2026-08-26). The run's output is committed in `outputs/backtest_results.csv` and
`outputs/performance_metrics.txt`, so every number below can be checked. (That metrics file was
written before the labels were corrected: its "Training Window: 5 years" means the 1,260-calendar-day
window described above, and "Probability Threshold: 0.45" means "tuned per training window".) Re-running on newer data
will move these numbers a little. The
baseline is the naive rule "always predict up".

| Measure | Model | Always "up" |
|---|---|---|
| Accuracy, all 5,538 daily predictions | 53.1% | 55.4% |
| Accuracy, 264 non-overlapping predictions (every 21st trading day) | 51.1% | 55.7% |
| ROC-AUC (0.5 = random ranking) | 0.48 (0.49 non-overlapping) | - |
| Years the model beats "always up" | 11 of 23 | - |
| Years with ROC-AUC above 0.5 | 5 of 23 | - |

How to read this:

- Daily predictions of a 21-day move overlap, so the 5,538 rows hold only about 264 independent
  outcomes. On those (starting from the first row) the model was right 135 times; the chance of
  doing at least that well by coin-flipping is about 38%. Across all 21 possible starting days the
  non-overlapping accuracy ranges from 49.4% to 56.8%, and it beats "always up" on only 1 of the 21
  (one more is a tie).
- Copper rose in about 55% of 21-day windows, so a model has to beat 55%, not 50%, to add anything.
- The walk-forward has no gap between the training and test windows, so the last training labels
  overlap the test period. That is expected to make these results look better, not worse.
- The tuned threshold sits at its lowest allowed value (0.35) for 30% of predictions, and the model
  predicts "up" 71.5% of the time - so its accuracy largely tracks the "always up" rule.
- This is the evidence available. It does not show a working predictor; it does not rule out that a
  different horizon, feature set or validation setup would do better.

Educational and research use only - not investment advice.

## Dashboard Views

1. **Overview**: Project description, metrics summary, download backtest results and performance metrics
2. **Backtest Explorer**: Charts, confusion matrix, threshold tuning, download filtered results
3. **Feature Importance**: XGBoost importances, download feature importance data (the SHAP section is a placeholder)
4. **Recent Predictions**: Last 30 days with probabilities, download recent predictions
5. **Settings**: horizon selection, threshold adjustment (the Retrain button is a placeholder - retrain with `python copper_brain_v2.py`)

## Download Features

The dashboard includes comprehensive download functionality to export data to your computer:

- **Backtest Results**: Complete historical predictions with probabilities and outcomes (CSV)
- **Performance Metrics**: Detailed model performance report (TXT)
- **Summary Report**: Key metrics in tabular format (CSV)
- **Filtered Results**: Export filtered backtest data based on your selected criteria (CSV)
- **Feature Importance**: Full feature rankings and category breakdowns (CSV)
- **Recent Predictions**: Last 30 days of predictions with price data (CSV)

All downloads include timestamps in the filename for easy organization.

## License

MIT License - Use at your own risk for educational purposes only.
