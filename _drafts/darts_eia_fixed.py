import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from darts import TimeSeries
from darts.models import ARIMA, Theta

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family='serif')




@dataclass
class Config:
    csv_path: str = (
        "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    )
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5


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
        # Take the first `horizon` timestamps from the future using DateTimeIndex
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


def maybe_load_tbats_csv():
    p = Path("eia_preds_tbats.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def main(plot: bool = False):
    cfg = Config()
    ts = load_series(cfg)

    results = {}
    preds = {}

    mean_mae, (y_true, y_pred) = rolling_origin_eval(
        ts, lambda: ARIMA(p=1, d=1, q=1), cfg.horizon, cfg.n_splits
    )
    results["ARIMA mean MAE"] = mean_mae
    preds["ARIMA"] = (y_true, y_pred)

    mean_mae, (y_true, y_pred) = rolling_origin_eval(
        ts, lambda: Theta(), cfg.horizon, cfg.n_splits
    )
    results["Theta mean MAE"] = mean_mae
    preds["Theta"] = (y_true, y_pred)

    tbats_df = maybe_load_tbats_csv()
    if tbats_df is not None and not tbats_df.empty:
        tbats_mae = mean_absolute_error(tbats_df["true"], tbats_df["pred"])
        results["TBATS mean MAE (from CSV)"] = tbats_mae

    logger.info("\n".join(f"{k}: {v}" for k, v in results.items()))

    # Tufte-style final figure: 2024 history, dashed vline at Jan 2025, forecasts/actuals Jan–Aug 2025 only
    start_2024 = pd.Period("2024-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period("2024-12", freq="M").start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period("2025-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period("2025-08", freq="M").start_time + pd.offsets.MonthBegin(0)

    s = ts.to_series()
    y_hist = s.loc[start_2024:end_2024]
    y_act = s.loc[jan_2025:aug_2025]

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
        ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
        if len(y_act):
            ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)

    # Collect forecasts restricted to Jan–Aug 2025
        end_labels = []
        for name, (yt, yp) in preds.items():
            f_idx = yp.time_index
            mask = (f_idx >= jan_2025) & (f_idx <= aug_2025)
            if mask.any():
                f = pd.Series(yp.values().ravel()[mask], index=f_idx[mask])
                ax.plot(f.index, f.values, color="#000000", lw=2.0, alpha=0.85)
                end_labels.append((f.index[-1], f.values[-1], name))

        if tbats_df is not None and not tbats_df.empty:
            f_tb = tbats_df[(tbats_df["date"] >= jan_2025) & (tbats_df["date"] <= aug_2025)]
            if not f_tb.empty:
                ax.plot(f_tb["date"], f_tb["pred"], color="#000000", lw=1.6, alpha=0.6)
                end_labels.append((f_tb["date"].iloc[-1], f_tb["pred"].iloc[-1], "TBATS"))

    # Minimal y-axis
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter

        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_xlabel("")

    # End-of-line labels
        if len(y_hist):
            ax.annotate(
                "History (2024)",
                xy=(y_hist.index[-1], y_hist.values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                ha="left",
                color="#666666",
            )
        if len(y_act):
            ax.annotate(
                "Actual (Jan–Aug 2025)",
                xy=(y_act.index[-1], y_act.values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                ha="left",
                color="#444444",
            )
        for x, yv, name in end_labels:
            ax.annotate(
                name,
                xy=(x, yv),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                ha="left",
                color="#000000",
            )

        ax.set_title(
            "EIA Net Generation — ARIMA/Theta/TBATS last-fold forecasts Jan–Aug 2025"
        )
        signalplot.save("eia_darts_tbats_last_fold.png")

    # Save ARIMA/Theta last fold predictions for reproducibility
    rows = []
    for name, (yt, yp) in preds.items():
        d = pd.DataFrame(
            {
                "model": name,
                "date": yt.time_index,
                "true": yt.values().ravel(),
                "pred": yp.values().ravel(),
            }
        )
        rows.append(d)
    pd.concat(rows).to_csv("eia_preds_darts.csv", index=False)


if __name__ == "__main__":
    main()
