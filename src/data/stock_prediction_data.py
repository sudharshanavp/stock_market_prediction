from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def data_input(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def sequential_window_dataset(series: pd.Series, window_size: int) -> pd.Series:
    """Create a sequential training dataset with the specified window size

    Args:
        series (pd.Series): pandas series of input features
        window_size (int): number of days

    Returns:
        pd.Series: sequential dataset with number of days
    """
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)


def data_train_test_split(dataframe: pd.DataFrame, split_dates):
    """Split the dataset into training, validation and testing data from the given dates

    Args:
        path (str): path of the input dataset
        split_dates (Tuple(str)): Tuple of two dates to split the dataset

    Returns:
        Dataframes: train, test and validation dataframes
    """

    dataframe["Date"] = pd.to_datetime(dataframe["Date"])

    train_split_date = split_dates[0]
    test_split_date = split_dates[1]

    train = dataframe.loc[dataframe["Date"] <= train_split_date]["Close"]
    test = dataframe.loc[dataframe["Date"] >= test_split_date]["Close"]
    valid = dataframe.loc[
        (dataframe["Date"] < test_split_date) & (dataframe["Date"] > train_split_date)
    ]["Close"]

    return train, valid, test


def data_normalization(
    train_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame
):
    """Normalize the input data

    Args:
        train (pd.DataFrame): Training dataset
        valid (pd.DataFrame): Validation dataset
        test (pd.DataFrame): Testing dataset

    Returns:
        Dataframes: Return the noramlized datasets of train, validation and test
    """
    # Reshape values
    train_values = train_data.values.reshape(-1, 1)
    valid_values = validation_data.values.reshape(-1, 1)
    test_values = test_data.values.reshape(-1, 1)

    #  Create Scaler Object
    x_train_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit x_train values
    normalized_x_train = x_train_scaler.fit_transform(train_values)

    # Fit x_valid values
    normalized_x_valid = x_train_scaler.transform(valid_values)

    # Fit x_test values
    normalized_x_test = x_train_scaler.transform(test_values)

    return normalized_x_train, normalized_x_valid, normalized_x_test


class RandomForestPreprocessing:
    def __init__(self, stock_symbol, days_out, n, w):
        self.stock_symbol = stock_symbol
        self.days_out = days_out
        self.n = n
        self.w = w
        self.price_data = pd.read_csv(
            "../data/processed/stocks/nse_scraped/" + self.stock_symbol + ".csv"
        )

    def values_sort_price_change_calculation(self):

        self.price_data = self.price_data[
            [
                "Date",
                "Symbol",
                "Prev Close",
                "Open",
                "High",
                "Low",
                "Last",
                "Close",
                "VWAP",
                "Volume",
            ]
        ]

        self.price_data.sort_values(by=["Symbol", "Date"], inplace=True)
        self.price_data["change_in_price"] = self.price_data["Close"].diff()

    def row_symbol_change(self):

        mask = self.price_data["Symbol"] != self.price_data["Symbol"].shift(1)
        self.price_data["change_in_price"] = np.where(
            mask == True, np.nan, self.price_data["change_in_price"]
        )
        self.price_data[self.price_data.isna().any(axis=1)]

    def grouping_signal_flag(self):

        price_data_smoothed = self.price_data.groupby(["Symbol"])[
            ["Close", "Low", "High", "Open", "Volume"]
        ].transform(lambda x: x.ewm(span=self.days_out).mean())

        smoothed_df = pd.concat(
            [self.price_data[["Symbol", "Date"]], price_data_smoothed],
            axis=1,
            sort=False,
        )
        return smoothed_df

    def signal_flag(self, smoothed_df):
        smoothed_df["Signal_Flag"] = smoothed_df.groupby("Symbol")["Close"].transform(
            lambda x: np.sign(x.diff(self.days_out))
        )

        return smoothed_df

    def RSI(self):
        up_df, down_df = (
            self.price_data[["Symbol", "change_in_price"]].copy(),
            self.price_data[["Symbol", "change_in_price"]].copy(),
        )

        up_df.loc["change_in_price"] = up_df.loc[
            (up_df["change_in_price"] < 0), "change_in_price"
        ] = 0

        down_df.loc["change_in_price"] = down_df.loc[
            (down_df["change_in_price"] > 0), "change_in_price"
        ] = 0

        down_df["change_in_price"] = down_df["change_in_price"].abs()

        ewma_up = up_df.groupby("Symbol")["change_in_price"].transform(
            lambda x: x.ewm(span=self.n).mean()
        )
        ewma_down = down_df.groupby("Symbol")["change_in_price"].transform(
            lambda x: x.ewm(span=self.n).mean()
        )

        relative_strength = ewma_up / ewma_down

        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        self.price_data["down_days"] = down_df["change_in_price"]
        self.price_data["up_days"] = up_df["change_in_price"]
        self.price_data["RSI"] = relative_strength_index

    def Stochastic_Oscillator(self):
        low_14, high_14 = (
            self.price_data[["Symbol", "Low"]].copy(),
            self.price_data[["Symbol", "High"]].copy(),
        )

        low_14 = low_14.groupby("Symbol")["Low"].transform(
            lambda x: x.rolling(window=self.n).min()
        )
        high_14 = high_14.groupby("Symbol")["High"].transform(
            lambda x: x.rolling(window=self.n).max()
        )

        k_percent = 100 * ((self.price_data["Close"] - low_14) / (high_14 - low_14))

        self.price_data["low_14"] = low_14
        self.price_data["high_14"] = high_14
        self.price_data["k_percent"] = k_percent

    def Williams(self):
        # Make a copy of the high and low column.
        low_14, high_14 = (
            self.price_data[["Symbol", "Low"]].copy(),
            self.price_data[["Symbol", "High"]].copy(),
        )

        low_14 = low_14.groupby("Symbol")["Low"].transform(
            lambda x: x.rolling(window=self.n).min()
        )
        high_14 = high_14.groupby("Symbol")["High"].transform(
            lambda x: x.rolling(window=self.n).max()
        )

        r_percent = ((high_14 - self.price_data["Close"]) / (high_14 - low_14)) * -100

        self.price_data["r_percent"] = r_percent

    def MACD(self):
        ema_26 = self.price_data.groupby("Symbol")["Close"].transform(
            lambda x: x.ewm(span=26).mean()
        )
        ema_12 = self.price_data.groupby("Symbol")["Close"].transform(
            lambda x: x.ewm(span=12).mean()
        )
        macd = ema_12 - ema_26

        ema_9_macd = macd.ewm(span=9).mean()

        self.price_data["MACD"] = macd
        self.price_data["MACD_EMA"] = ema_9_macd

    def rate_of_change(self):
        self.price_data["Price_Rate_Of_Change"] = self.price_data.groupby("Symbol")[
            "Close"
        ].transform(lambda x: x.pct_change(periods=self.w))

    def obv(self, group):

        volume = group["Volume"]
        change = group["Close"].diff()

        prev_obv = 0
        obv_values = []

        for i, j in zip(change, volume):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            # OBV.append(current_OBV)
            prev_obv = current_obv
            obv_values.append(current_obv)

        # Return a panda series.
        return pd.Series(obv_values, index=group.index)

    def apply_obv_to_groups(self):
        obv_groups = self.price_data.groupby("Symbol").apply(self.obv)

        self.price_data["On Balance Volume"] = obv_groups.reset_index(
            level=0, drop=True
        )

    def close_group(self):
        close_groups = self.price_data.groupby("Symbol")["Close"]
        close_groups = close_groups.transform(lambda x: np.sign(x.diff()))
        self.price_data["Prediction"] = close_groups
        self.price_data.loc[self.price_data["Prediction"] == 0.0] = 1.0

    def close_group_nan_remove(self):
        print(
            "Before NaN Drop we have {} rows and {} columns".format(
                self.price_data.shape[0], self.price_data.shape[1]
            )
        )

        self.price_data = self.price_data.dropna()

        print(
            "After NaN Drop we have {} rows and {} columns".format(
                self.price_data.shape[0], self.price_data.shape[1]
            )
        )
