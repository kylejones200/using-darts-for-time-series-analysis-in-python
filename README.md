# Using Darts for Time Series Analysis in Python

This project demonstrates time series forecasting using the Darts library, including ARIMA, Exponential Smoothing, LightGBM, LSTM, NBEATS, and FFT models.

## Business context

Economists and investors watch the **10Y–2Y Treasury spread (T10Y2Y)** because it compresses expectations about growth and policy. Inversion has often preceded recessions, so teams want comparable forecasts—not just a chart—with clear train/holdout discipline.

**Darts** wraps statistical and ML forecasters behind one `TimeSeries` workflow so you can benchmark ARIMA, exponential smoothing, LightGBM, FFT, and optional deep models on the same FRED series without rewriting glue code each time.

## Article

Medium article: [Using Darts for Time Series Analysis in Python](https://medium.com/@kylejones_47003/using-darts-for-time-series-analysis-in-python-dc92e08c43e5)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series ID or local file path)
- Model parameters (ARIMA order, LSTM epochs, etc.)
- Which models to run
- Output settings

FRED series are loaded via **pandas_datareader** (no API key required).

## Caveats

- Deep learning models (LSTM, NBEATS) are disabled by default in config.yaml due to longer training times. Enable them by setting `enabled: true`.
- Data is automatically split before scaling to prevent data leakage.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).