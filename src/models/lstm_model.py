import data.stock_prediction_data as StockData
import tensorflow as tf
import numpy as np


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


class LongShortTermMemory:
    def __init__(self, path: str) -> None:
        self.input_df = StockData.data_input(path)
        self.split_dates = ("2014-12-31", "2019-01-02")
        self.train_df, self.validate_df, self.test_df = StockData.data_train_test_split(
            self.input_df, self.split_dates
        )
        self.train, self.validate, self.test = StockData.data_normalization(
            self.train_df, self.validate_df, self.test_df
        )

    def create_model(self):
        keras = tf.keras
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        # create model
        model = keras.models.Sequential(
            [
                keras.layers.LSTM(
                    100,
                    return_sequences=True,
                    stateful=True,
                    batch_input_shape=[1, None, 1],
                ),
                keras.layers.LSTM(100, return_sequences=True, stateful=True),
                keras.layers.Dense(1),
            ]
        )
        return model

    def optimal_learning_rate(self, window_size: int):
        keras = tf.keras
        model = self.create_model()

        train_set = StockData.sequential_window_dataset(self.train, window_size)

        # create lr
        lr_schedule = keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-5 * 10 ** (epoch / 20)
        )
        reset_states = ResetStatesCallback()

        # choose optimizer
        optimizer = keras.optimizers.Nadam(lr=1e-5)

        # compile model
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

        # set history
        history = model.fit(
            train_set, epochs=100, callbacks=[lr_schedule, reset_states]
        )

        return history.history

    def train_model(self, window_size: int, learning_rate: float, save_path: str):
        """train the LSTM model using train dataset

        Args:
            window_size (int): number of days
            train (_type_): training dataset in pandas dataframe
        """
        keras = tf.keras
        # set window size and create input batch sequence
        train_set = StockData.sequential_window_dataset(self.train, window_size)
        valid_set = StockData.sequential_window_dataset(self.validate, window_size)

        model = self.create_model()

        # set optimizer
        optimizer = keras.optimizers.Nadam(lr=learning_rate)

        # compile model
        model.compile(
            loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae", "accuracy"]
        )

        # reset states
        reset_states = ResetStatesCallback()

        # set up save best only checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True
        )

        early_stopping = keras.callbacks.EarlyStopping(patience=50)

        # fit model
        model.fit(
            train_set,
            epochs=100,
            validation_data=valid_set,
            callbacks=[early_stopping, model_checkpoint, reset_states],
            verbose=0,
        )

        return model

    def test_model(self, model_path: str):
        keras = tf.keras
        model = keras.models.load_model(model_path)
        lstm_forecast = model.predict(self.test[np.newaxis, :])
        model.evaluate(self.test[np.newaxis, :], lstm_forecast)
