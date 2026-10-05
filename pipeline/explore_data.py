"""Quick data-access sanity check for the DSM500 proposal.
Pulls a little crypto (ccxt) and an index (yfinance) and prints/saves a sample.
Run: source .venv/bin/activate && python explore_data.py
"""
from pathlib import Path
import pandas as pd

OUT = Path('data'); OUT.mkdir(exist_ok=True)

def crypto_hourly(symbol='BTC/USDT', limit=500):
    """Download hourly crypto candles through ccxt for exploration."""
    import ccxt
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('ts')

def index_daily(ticker='^GSPC', period='5y'):
    """Download daily closes for a ticker through yfinance for exploration."""
    import yfinance as yf
    return yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)

if __name__ == '__main__':
    try:
        btc = crypto_hourly()
        print('crypto hourly:', btc.shape)
        print(btc.tail(3))
        btc.to_parquet(OUT / 'btc_1h.parquet')
    except Exception as e:
        print('crypto fetch failed:', e)
    try:
        spx = index_daily()
        print('index daily:', spx.shape)
        spx.to_parquet(OUT / 'spx_1d.parquet')
    except Exception as e:
        print('index fetch failed:', e)
