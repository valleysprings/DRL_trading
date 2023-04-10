from __future__ import annotations

import datetime
from datetime import timedelta
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

import config
from processor.yahoodownloader import YahooDownloader



def data_split(df, start, end, target_date_col="timestamp"):
    """
    split the dataset into training or testing using timestamp
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class FeatureEngineer:
    """Provides methods for preprocessing the crypto price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names
    Methods
    -------
    preprocess_data()
        main method to do the feature engineering
    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list

    def preprocess_data(self, df, args):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        start, end, freq = args
        df = self.clean_data(df, start, end, freq)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data, start, end, freq):
        """
        clean the raw data
        deal with missing values
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp", "tic"], ignore_index=True)
        df.index = df.timestamp.factorize()[0]
        merged_closes = df.pivot_table(index="timestamp", columns="tic", values="close")
        # print(merged_closes)
        
        
        time = pd.DataFrame()
        start = pd.to_datetime(start) + pd.Timedelta(freq)
        end = pd.to_datetime(end) + pd.Timedelta(freq)
        time["timestamp"]=pd.DataFrame(pd.date_range(start=start, end=end, freq=freq))
        
        merged_closes.index = pd.to_datetime(merged_closes.index)
        res = pd.merge(merged_closes, time, how='right', on="timestamp")
        revert = res.melt(id_vars="timestamp", var_name = "tic" ,value_name= "close", value_vars=df.tic.unique())
        df = pd.merge(df, revert, how='right', on=["timestamp", "tic"])
        
        del df["close_x"]
        df = df.rename(columns={"close_y": "close"})
        df.fillna(method="ffill", axis=0,inplace=True)
        df.fillna(method="bfill", axis=0,inplace=True)
        
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]], on=["tic", "timestamp"], how="left"
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df
