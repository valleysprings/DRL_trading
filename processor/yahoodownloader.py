"""Contains methods and classes to collect data from
Yahoo Finance API
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import timedelta

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data 
        end_date : str
            end date of the data 
        ticker_list : list
            a list of stock tickers  
        time_interval : str
            time interval between each data point

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list, time_interval: str):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.time_interval = time_interval

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: timestamp, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """

        
        start_date = pd.Timestamp(self.start_date)
        end_date = pd.Timestamp(self.end_date)
        data_df = pd.DataFrame()
        temp_df = pd.DataFrame()
        
        if self.time_interval == "1d":

            full_df = yf.download(
                self.ticker_list,
                start=start_date,
                end=end_date,
                interval=self.time_interval,
            )
            
            if len(self.ticker_list) == 1:
                full_df["tic"] = self.ticker_list[0]
                data_df = full_df
            else:
                full_df.columns = full_df.columns.swaplevel(0, 1)
                full_df.sort_index(axis = 1, level = 0, inplace = True)
                
                for tic in self.ticker_list:
                    temp_df = full_df[tic]
                    temp_df["tic"] = tic
                    data_df = data_df.append(temp_df)
                    temp_df = temp_df.dropna()

                
        elif self.time_interval == "1h":
            delta = timedelta(days = 300)

            while (
                start_date < end_date
            ):  # downloading daily to workaround yfinance only allowing  max 7 calendar (not trading) days of 1 min data per single download
                if start_date + delta <= end_date:
                    delta = end_date - start_date
                    
                full_df = yf.download(
                    self.ticker_list,
                    start=start_date,
                    end=start_date + delta,
                    interval=self.time_interval,
                )
                
                if len(self.ticker_list) == 1:
                    full_df["tic"] = self.ticker_list[0]
                    data_df = full_df
                else:
                    full_df.columns = full_df.columns.swaplevel(0, 1)
                    full_df.sort_index(axis = 1, level = 0, inplace = True)                
                    
                    for tic in self.ticker_list:
                        temp_df = full_df[tic]
                        temp_df["tic"] = tic
                        data_df = data_df.append(temp_df)
                        temp_df = temp_df.dropna()
                start_date += delta
                
        else:
            delta = timedelta(days = 7)

            while (
                start_date < end_date
            ):  # downloading daily to workaround yfinance only allowing  max 7 calendar (not trading) days of 1 min data per single download
                if start_date + delta <= end_date:
                    delta = end_date - start_date
                
                full_df = yf.download(
                    self.ticker_list,
                    start=start_date,
                    end=start_date + delta,
                    interval=self.time_interval,
                )
                
                if len(self.ticker_list) == 1:
                    full_df["tic"] = self.ticker_list[0]
                    data_df = full_df
                else:
                    full_df.columns = full_df.columns.swaplevel(0, 1)
                    full_df.sort_index(axis = 1, level = 0, inplace = True)                
                    
                    for tic in self.ticker_list:
                        temp_df = full_df[tic]
                        temp_df["tic"] = tic
                        data_df = data_df.append(temp_df)
                        temp_df = temp_df.dropna()
                start_date += delta
                
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            print(data_df.columns)
            # convert the column names to standardized names
            data_df.columns = [
                "timestamp",
                "adjcp",
                "close",
                "high",
                "low",
                "open",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")

        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        data_df = data_df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)

        return data_df
