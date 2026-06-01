#!/usr/bin/env python3
"""
Time Series Forecasting with Darts

Main entry point for running forecasting models.
"""

import argparse
import logging
from pathlib import Path

import yaml
from src.core import (
    evaluate_forecast,
    fit_arima,
    fit_exponential_smoothing,
    fit_fft,
    fit_lightgbm,
    fit_lstm,
    fit_nbeats,
    forecast_model,
    load_data,
    plot_forecast,
    prepare_data,
)


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Time Series Forecasting with Darts")
    parser.add_argument("--config", type=Path, default=None, help="Path to config file")
    parser.add_argument(
        "--data-path", type=Path, default=None, help="Path to data file"
    )
    parser.add_argument("--api-key", type=str, default=None, help="FRED API key")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for plots"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)
    data_path = (
        args.data_path
        if args.data_path
        else (Path(config["data"]["source"]) if config["data"]["source"] else None)
    )
    series_id = config["data"]["series_id"]
    if data_path:
        series = load_data(data_path=data_path)
    elif series_id:
        series = load_data(series_id=series_id)
    else:
        raise ValueError("Must provide either --data-path or series_id in config")

    train, val, scaler = prepare_data(series, config["data"]["split_date"])
    logging.info(f"Training: {len(train)} observations")
    logging.info(f"Validation: {len(val)} observations")
    models_config = config["models"]

    def run_model(name: str, model, plot_name: str, plot_file: str) -> None:
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        logging.info(f"{name} - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(
            series, forecast, plot_name, output_dir / plot_file, metrics
        )

    if models_config["arima"]["enabled"]:
        model = fit_arima(train, tuple(models_config["arima"]["order"]))
        run_model("ARIMA", model, "ARIMA Forecast", "arima_forecast.png")

    if models_config["exponential_smoothing"]["enabled"]:
        model = fit_exponential_smoothing(train)
        run_model(
            "Exponential Smoothing",
            model,
            "Exponential Smoothing Forecast",
            "exponential_smoothing_forecast.png",
        )

    if models_config["lightgbm"]["enabled"]:
        model = fit_lightgbm(train, models_config["lightgbm"]["lags"])
        run_model("LightGBM", model, "LightGBM Forecast", "lightgbm_forecast.png")

    if models_config["lstm"]["enabled"]:
        model = fit_lstm(
            train,
            val,
            models_config["lstm"]["input_chunk_length"],
            models_config["lstm"]["output_chunk_length"],
            models_config["lstm"]["n_epochs"],
        )
        run_model("LSTM", model, "LSTM Forecast", "lstm_forecast.png")

    if models_config["nbeats"]["enabled"]:
        model = fit_nbeats(
            train,
            val,
            models_config["nbeats"]["input_chunk_length"],
            models_config["nbeats"]["output_chunk_length"],
            models_config["nbeats"]["n_epochs"],
            models_config["nbeats"]["random_state"],
        )
        run_model("NBEATS", model, "NBEATS Forecast", "nbeats_forecast.png")

    if models_config["fft"]["enabled"]:
        model = fit_fft(train)
        run_model("FFT", model, "FFT Forecast", "fft_forecast.png")

    logging.info(f"\nForecasting complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
