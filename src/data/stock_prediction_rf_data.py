import pandas as pd
import numpy as np


def change_in_price(in_data: pd.DataFrame):

    new_data = in_data[
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

    new_data = new_data.sort_values(by=["Symbol", "Date"])
    new_data = new_data["change_in_price"] = new_data["Close"].diff()


def symbol_change_row(in_data: pd.DataFrame):
    mask = in_data["Symbol"] != in_data["Symbol"].shift(1)
    in_data["change_in_price"] = np.where(
        mask == True, np.nan, in_data["change_in_price"]
    )
    in_data[in_data.isna().any(axis=1)]
    return in_data


class RandomForestPreprocessing:
    def __init__(self, stock_symbol, days_out, n, w):
        self.stock_symbol = stock_symbol
        self.days_out = days_out
        self.n = n
        self.w = w
        self.price_data = pd.read_csv(
            "../data/processed/stocks/nse_scraped/" + self.stock_symbol + ".csv"
        )
        self.features = pd.DataFrame()

    # def values_sort_price_change_calculation(self):

    #     self.price_data = self.price_data[
    #         [
    #             "Date",
    #             "Symbol",
    #             "Prev Close",
    #             "Open",
    #             "High",
    #             "Low",
    #             "Last",
    #             "Close",
    #             "VWAP",
    #             "Volume",
    #         ]
    #     ]

    #     self.price_data
    #     self.price_data["change_in_price"] = self.price_data["Close"].diff()

    # def row_symbol_change(self):

    #     mask = self.price_data["Symbol"] != self.price_data["Symbol"].shift(1)
    #     self.price_data["change_in_price"] = np.where(
    #         mask == True, np.nan, self.price_data["change_in_price"]
    #     )
    #     self.price_data[self.price_data.isna().any(axis=1)]

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
