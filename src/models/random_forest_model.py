import pandas as pd
from functools import reduce
import data.stock_prediction_rf_data as StockData

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score


class RandomForest:
    def __init__(self, in_data, days) -> None:
        self.in_data = StockData.string_to_int(StockData.return_dataframe(in_data))
        self.features_data = pd.DataFrame()
        self.days = days
        self.target_names = ["Down Day", "Up Day"]

    def feature_engineering(self):

        price_diff_feature = StockData.change_in_price(self.in_data)
        rsi_feature = StockData.relative_strength_index(self.days, price_diff_feature)
        stoch_osci_feature = StockData.stochastic_oscillator(
            self.days, self.in_data[["Symbol", "Low", "High", "Close"]]
        )
        williams_feature = StockData.williams(
            self.days, self.in_data[["Symbol", "Low", "High", "Close"]]
        )
        macd_feature = StockData.macd(self.in_data[["Symbol", "Close"]])
        proc_feature = StockData.price_rate_of_change(9, self.in_data)
        
        self.features_data["RSI"] = pd.Series(rsi_feature).values
        self.features_data["k_percent"] = pd.Series(stoch_osci_feature).values
        self.features_data["r_percent"] = pd.Series(williams_feature).values
        self.features_data["MACD"] = pd.Series(macd_feature).values
        self.features_data["Price_Rate_Of_Change"] = pd.Series(proc_feature).values
        
        obv_feature = StockData.apply_obv(self.in_data)
        target = StockData.create_prediction_column(self.in_data[["Symbol", "Close"]])

        self.features_data["On Balance Volume"] = pd.Series(obv_feature).values
        self.features_data["Prediction"] = pd.Series(target).values

        self.features_data = StockData.remove_null_values(self.features_data)

    def random_search(self):
        n_estimators = list(range(200, 2000, 200))
        max_features = ["auto", "sqrt", None, "log2"]
        max_depth = list(range(10, 110, 10))
        max_depth.append(None)
        min_samples_split = [2, 5, 10, 20, 30, 40]
        min_samples_leaf = [1, 2, 7, 12, 14, 16, 20]
        bootstrap = [True, False]
        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap
        }

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )

        return rf_random

    def train_model(
        self,
        rf_model=RandomForestClassifier(
            n_estimators=100, oob_score=True, criterion="gini", random_state=0
        ),
    ):

        x_train, _, y_train, _ = StockData.split_data(self.features_data)

        rf_model.fit(x_train, y_train)

        return rf_model

    def predict_result(self, rf_model, feature=pd.DataFrame()):
        if feature.empty:
            feature = self.features_data.tail(self.days)
        return rf_model.predict(feature)

    def test_model(self, rf_model):

        _, _, _, y_test = StockData.split_data(self.features_data)

        y_pred = self.predict_result(rf_model, y_test)

        accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100.0

        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            target_names=self.target_names,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        return accuracy, report_df
