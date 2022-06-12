from typing import Tuple
import pandas as pd
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
