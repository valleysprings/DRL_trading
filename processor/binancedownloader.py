import pandas as pd
from binance import Client
from datetime import datetime, timedelta
from typing import List

key = "0256KLUp4O9fNv7eSq0cuj0d947WQT8vMBwNS5GzZQ7r1YFzKLqtJjlikIvNUCjr"
secret_key = "TIzrASsXyGEX8Ww3TWlac1739begTcxKZV690pSRowDvI0shEy5YdsNAXwzQYHad"

client = Client(key, secret_key)


class BinanceDownloader:
    def __init__(self) -> None:
        self.time_frames = [
            '1m',
            '2m',
            '3m',
            '5m',
            '15m',
            '30m',
            '1h',
            '2h',
            '4h',
            '6h',
            '8h',
            '12h',
            '1d',
            '3d',
            '1w',
            '1M'
        ]

    def get_data(self, start_date: str, end_date: str, ticker: str, ts=str) -> pd.DataFrame():
        return self._download_data(start_date=start_date, end_date = end_date, ticker=ticker, ts=ts)

    def get_datas(self, start_date: str, end_date: str, tickers: List, ts=str) -> pd.DataFrame():
        datas = [self._download_data(start_date=start_date, end_date = end_date, ticker=tic, ts=ts)
                 for tic in tickers]
        return pd.concat(datas, axis=0)

    def _download_data(
        self,
        start_date: str = '2021-1-1',
        end_date: str = '2023-3-1',
        ticker: str = 'BTCUSDT',
        ts: str = '1h'
    ) -> pd.DataFrame():
        if ts not in self.time_frames:
            raise ValueError(f'Support time frames: {self.time_frames}')
        
        # start_date = pd.to_datetime(start_date)
        # end_date = pd.to_datetime(end_date)
            
        klines = client.get_historical_klines(
            ticker, ts, start_date, end_date
        )
        data = self._convert_to_dataframe(klines)
        data['tic'] = ticker
        return data

    def _convert_to_dataframe(
        self,
        klines: List
    ) -> pd.DataFrame():
        data = pd.DataFrame(
            data=[row[1:7] for row in klines],
            columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'],
        ).set_index('timestamp')
        data.index = pd.to_datetime(data.index + 1, unit='ms')
        data = data.sort_index()
        data = data.apply(pd.to_numeric, axis=1)
        return data

