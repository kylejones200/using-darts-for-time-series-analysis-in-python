import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from darts import TimeSeries
from darts.models import ExponentialSmoothing, NaiveSeasonal
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "data/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> TimeSeries:
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError("EIA CSV not found")
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    s = TimeSeries.from_dataframe(
        df, time_col="date", value_cols="value", freq=cfg.freq
    ).astype(float)
    return s


def rolling_origin_eval(ts: TimeSeries, model_ctor, horizon: int, n_splits: int):
    values = ts.values().ravel()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(len(values))
    maes = []
    last_true, last_pred = None, None
    for train_idx, test_idx in tscv.split(idx):
        end = train_idx[-1]
        cutoff = ts.time_index[end]
        y_tr, future = ts.split_after(cutoff)
        if len(future) == 0:
            continue
        end_idx = min(horizon, len(future)) - 1
        y_te = future.drop_after(future.time_index[end_idx])
        m = model_ctor()
        m.fit(y_tr)
        fc = m.predict(len(y_te))
        mae = mean_absolute_error(y_te.values().ravel(), fc.values().ravel())
        maes.append(mae)
        last_true, last_pred = y_te, fc
    return np.mean(maes), (last_true, last_pred)


def main(plot: bool = False):
    cfg = Config()
    ts = load_series(cfg)

    results = {}
    preds = {}

    mean_mae, (y_true, y_pred) = rolling_origin_eval(
        ts,
        lambda: ExponentialSmoothing(seasonal_periods=cfg.season),
        cfg.horizon,
        cfg.n_splits,
    )
    results["ExponentialSmoothing mean MAE"] = mean_mae
    preds["ETS"] = (y_true, y_pred)

    mean_mae, (y_true, y_pred) = rolling_origin_eval(
        ts, lambda: NaiveSeasonal(K=cfg.season), cfg.horizon, cfg.n_splits
    )
    results["NaiveSeasonal mean MAE"] = mean_mae
    preds["NaiveSeasonal"] = (y_true, y_pred)

    logger.info("\n".join(f"{k}: {v}" for k, v in results.items()))

    if plot:
        plt.figure(figsize=(9, 4))
        ts.plot(label="history", alpha=0.6)
        for name, (yt, yp) in preds.items():
            yp.plot(label=f"{name} last fold")
        plt.legend()
        signalplot.save("eia_darts_overview_last_fold.png")


if __name__ == "__main__":
    main()
