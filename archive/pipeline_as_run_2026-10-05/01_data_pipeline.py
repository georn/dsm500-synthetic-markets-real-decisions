#!/usr/bin/env python3
"""
Data Pipeline for DSM500 CW2: Synthetic Markets, Real Decisions

Collects, cleans, and windows financial time-series data for generative model training
and strategy validation. Outputs two eras (train/dev and held-out test) as Parquet.

Data sources:
- BTC/USDT, ETH/USDT: hourly via ccxt (Coinbase)
- EUR/USD, SPY, ^FTSE: daily via yfinance
- Period: 2018-01-01 to 2024-12-31
- Era split: 2018-2021 (training), 2022-2024 (held-out test)

Output:
- outputs/windowed_data_era_a.parquet (training windows)
- outputs/windowed_data_era_b.parquet (test windows)
- outputs/data_metadata.json (collection dates, record counts, diagnostics)

Usage:
    python 01_data_pipeline.py [--skip-download] [--output-dir ./outputs]

Reproducibility:
- Fixed random seed (42)
- All timestamps UTC
- Overlapping windows with stride documented
- Chronological split applied BEFORE windowing to prevent look-ahead leakage
- Transaction costs and slippage modeled separately (not here)
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
from dotenv import load_dotenv

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)

# Configuration
CONFIG = {
    "era_a_start": "2018-01-01",
    "era_a_end": "2021-12-31",
    "era_b_start": "2022-01-01",
    "era_b_end": "2024-12-31",
    "window_length_hourly": 256,  # ~11 days of hourly returns
    "window_stride_hourly": 24,   # 1-day stride (overlapping)
    "window_length_daily": 256,   # ~1 year of daily returns
    "window_stride_daily": 1,     # 1-day stride
}


class DataCollector:
    """Collect market data from multiple sources."""

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}

    def get_crypto_hourly(self, symbol="BTC/USDT", limit=None):
        """Download hourly crypto data via ccxt (Coinbase)."""
        logger.info(f"Downloading {symbol} hourly data via ccxt...")
        try:
            exchange = ccxt.coinbase()
            timeframe = "1h"
            all_candles = []

            since = exchange.parse8601("2018-01-01T00:00:00Z")
            end_time = exchange.parse8601("2024-12-31T23:59:59Z")

            while since < end_time:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=300)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1000  # Next second after last candle
                    logger.info(f"  Fetched {len(candles)} candles, total: {len(all_candles)}")
                except Exception as e:
                    logger.warning(f"  Rate limit or error: {e}, pausing...")
                    import time
                    time.sleep(2)
                    continue

            if not all_candles:
                raise ValueError(f"No data retrieved for {symbol}")

            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()

            logger.info(f"  Collected {len(df)} hourly {symbol} candles")
            self.metadata[f"{symbol}_hourly"] = {
                "source": "ccxt (Coinbase)",
                "records": len(df),
                "date_range": [df.index.min().isoformat(), df.index.max().isoformat()]
            }
            return df
        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}")
            return None

    def get_daily_data(self, ticker, name=None):
        """Download daily data via yfinance."""
        name = name or ticker
        logger.info(f"Downloading {name} ({ticker}) daily data via yfinance...")
        try:
            df = yf.download(
                ticker,
                start="2018-01-01",
                end="2024-12-31",
                progress=False
            )
            df.index = pd.to_datetime(df.index, utc=True)
            # yfinance returns MultiIndex columns: select 'Close' column for this ticker
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]  # Get the Close level, drops other price columns
                if isinstance(df, pd.DataFrame):  # If multiple tickers, take just this one
                    df = df[[ticker]]
                    df.columns = ["close"]
            else:
                df = df[["Close"]].rename(columns={"Close": "close"})
            df = df.sort_index()

            logger.info(f"  Collected {len(df)} daily {name} bars")
            self.metadata[f"{ticker}_daily"] = {
                "source": "yfinance",
                "records": len(df),
                "date_range": [df.index.min().isoformat(), df.index.max().isoformat()]
            }
            return df
        except Exception as e:
            logger.error(f"Failed to collect {ticker}: {e}")
            return None

    def compute_log_returns(self, df, col="close"):
        """Compute log returns from price series. Ensure 1D output."""
        returns = np.log(df[col] / df[col].shift(1))
        returns = returns.dropna()
        # Ensure returns is 1D (not 2D from DataFrame columns)
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]  # Take first column if DataFrame
        return returns


class DataWindower:
    """Create overlapping windows for training generative models."""

    @staticmethod
    def create_windows(returns, window_length, stride):
        """
        Create overlapping windows of returns.

        Args:
            returns: Series of returns
            window_length: Number of steps per window
            stride: Stride between windows

        Returns:
            2D array of shape (num_windows, window_length)
        """
        n = len(returns)
        windows = []
        for start in range(0, n - window_length + 1, stride):
            window = returns.iloc[start:start + window_length].values
            if len(window) == window_length:
                windows.append(window)
        return np.array(windows)

    @staticmethod
    def split_eras(df, era_a_end):
        """
        Split data chronologically into era-A (training) and era-B (test).
        Discard boundary-straddling windows to prevent leakage.
        """
        era_a = df[df.index <= era_a_end]
        era_b = df[df.index > era_a_end]
        return era_a, era_b


class DataPipeline:
    """Main pipeline: collect → clean → window → save."""

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = DataCollector(output_dir)
        self.metadata = {}

    def run(self, skip_download=False):
        """Execute full pipeline."""
        logger.info("=" * 80)
        logger.info("Starting DSM500 Data Pipeline")
        logger.info("=" * 80)

        # Step 1: Collect data
        logger.info("\n[Step 1] Collecting data...")
        data_raw = {}

        if not skip_download:
            # Crypto (hourly)
            for symbol in ["BTC/USDT", "ETH/USDT"]:
                data_raw[symbol] = self.collector.get_crypto_hourly(symbol)

            # Traditional (daily)
            for ticker, name in [("EUR=X", "EUR/USD"), ("SPY", "SPY"), ("^FTSE", "FTSE")]:
                data_raw[ticker] = self.collector.get_daily_data(ticker, name)
        else:
            logger.info("  Skipping download (--skip-download flag set)")

        # Remove None entries
        data_raw = {k: v for k, v in data_raw.items() if v is not None}

        if not data_raw:
            logger.error("No data collected. Aborting.")
            return False

        # Step 2: Compute log returns and era split
        logger.info("\n[Step 2] Computing log returns and splitting eras...")
        era_a_end = pd.Timestamp(CONFIG["era_a_end"], tz="UTC")
        era_b_start = pd.Timestamp(CONFIG["era_b_start"], tz="UTC")

        windower = DataWindower()
        era_a_windows = {}
        era_b_windows = {}

        for ticker, df in data_raw.items():
            if df is None or len(df) == 0:
                logger.warning(f"  Skipping {ticker} (empty)")
                continue

            # Compute returns
            returns = self.collector.compute_log_returns(df)
            logger.info(f"  {ticker}: {len(returns)} returns")

            # Split eras
            returns_a, returns_b = windower.split_eras(returns, era_a_end)
            logger.info(f"    Era-A: {len(returns_a)} returns | Era-B: {len(returns_b)} returns")

            # Determine window parameters
            if "USDT" in ticker or "=" in ticker:  # Crypto or FX (hourly or daily with hourly windows)
                if "USDT" in ticker:
                    window_length, stride = CONFIG["window_length_hourly"], CONFIG["window_stride_hourly"]
                else:
                    window_length, stride = CONFIG["window_length_daily"], CONFIG["window_stride_daily"]
            else:  # Traditional daily
                window_length, stride = CONFIG["window_length_daily"], CONFIG["window_stride_daily"]

            # Create windows
            windows_a = windower.create_windows(returns_a, window_length, stride)
            windows_b = windower.create_windows(returns_b, window_length, stride)

            logger.info(f"    Era-A windows: {len(windows_a)} | Era-B windows: {len(windows_b)}")

            era_a_windows[ticker] = windows_a
            era_b_windows[ticker] = windows_b

            self.metadata[ticker] = {
                "window_length": window_length,
                "stride": stride,
                "era_a_windows": len(windows_a),
                "era_b_windows": len(windows_b),
            }

        # Step 3: Convert to DataFrames and save
        logger.info("\n[Step 3] Saving windowed data to Parquet...")

        # Flatten and concatenate (each row = one window)
        era_a_list = []
        era_b_list = []
        for ticker in era_a_windows:
            for i, window in enumerate(era_a_windows[ticker]):
                row = {f"{ticker}_step_{j}": window[j] for j in range(len(window))}
                row["asset"] = ticker
                row["window_id"] = f"{ticker}_a_{i}"
                era_a_list.append(row)

            for i, window in enumerate(era_b_windows[ticker]):
                row = {f"{ticker}_step_{j}": window[j] for j in range(len(window))}
                row["asset"] = ticker
                row["window_id"] = f"{ticker}_b_{i}"
                era_b_list.append(row)

        df_era_a = pd.DataFrame(era_a_list)
        df_era_b = pd.DataFrame(era_b_list)

        era_a_path = self.output_dir / "windowed_data_era_a.parquet"
        era_b_path = self.output_dir / "windowed_data_era_b.parquet"

        df_era_a.to_parquet(era_a_path, compression="snappy")
        df_era_b.to_parquet(era_b_path, compression="snappy")

        logger.info(f"  Saved era-A: {era_a_path} ({len(df_era_a)} rows)")
        logger.info(f"  Saved era-B: {era_b_path} ({len(df_era_b)} rows)")

        # Step 4: Save metadata
        logger.info("\n[Step 4] Saving metadata...")
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "config": CONFIG,
            "data_summary": self.metadata,
            "era_a_shape": df_era_a.shape,
            "era_b_shape": df_era_b.shape,
        }

        metadata_path = self.output_dir / "data_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Saved metadata: {metadata_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Data Pipeline Complete")
        logger.info("=" * 80)
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSM500 Data Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download (use cached)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    pipeline = DataPipeline(output_dir=args.output_dir)
    success = pipeline.run(skip_download=args.skip_download)
    sys.exit(0 if success else 1)
