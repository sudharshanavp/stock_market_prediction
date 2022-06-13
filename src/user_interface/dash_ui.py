from dash import Dash, dcc, html, Input, Output
from plotly.express import data
import plotly.express as px
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import dash_daq as daq
import csv

# import urllib2
import requests
import pandas as pd

nifty50 = "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/ind_nifty50list.csv"
adani = "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/raw/stock/yahoo_finance/ADANIPORTS"
dff = pd.read_csv(nifty50)
df1 = pd.read_csv(adani)
symbol = pd.Series(dff["Symbol"])
symbol.index = pd.Series(dff["Company Name"])
symbol.to_dict()

# df =  pd.read_csv("companyNamesList.csv")
# url='https://github.com/sudharshanavp/stock_market_prediction/blob/machine_learning/data/ind_nifty50list.csv'
# df1 = pd.read_csv(url)
theme = {
    "dark": False,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div(
    [
        html.H1(children="Stock Market Prediction"),
        html.Div(
            [
                html.H2(children="Price Prediction"),
                dcc.Dropdown(
                    symbol.index, id="pandas-dropdown-1", value=dff["Company Name"][0]
                ),
                html.Div(id="output-container-1"),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [html.H2(children="Sentiment Analysis"), html.Div(id="output-container-2")],
            style={"width": "49%", "float": "right", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H2("Stock price analysis"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.RadioItems(
                                id="radioo",
                                options=["High", "Low", "Open", "Close"],
                                value="High",
                                labelStyle={"display": "block"},
                            ),
                            md=1,
                            style={"padding-left": "20px"},
                        ),
                        dbc.Col(dcc.Graph(id="time-series-chart"), md=10),
                    ],
                    align="center",
                ),
                html.P("Select stock:"),
                dcc.Dropdown(
                    id="ticker",
                    options=dff.Symbol,
                    value="AMZN",
                    clearable=False,
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("output-container-1", "children"),
        Output("output-container-2", "children"),
    ],
    Input("pandas-dropdown-1", "value"),
)
def update_output(value):
    # print(value)
    container1 = html.Div(
        [
            "Estimated Price: {}".format(value),
            html.Br(),
            "Prediction Model: LSTM 30 Day Moving Averrage ",
            html.Br(),
            "Accuracy of Model: 89.93 ",
            html.Br(),
            "Stock Trend: Downward",
        ]
    )
    container2 = html.Div(
        [
            "Overall Market Sentiment: Positive",
            html.Br(),
            "Stock Sentiment: Negative",
            html.Br(),
            "Sentimental Score: 89.93 ",
            html.Br(),
            "Accuracy of Sentiment Analysis: 95.6%",
        ]
    )
    return container1, container2


@app.callback(
    Output("time-series-chart", "figure"),
    [
        Input("ticker", "value"),
        Input("radioo", "value"),
        Input("pandas-dropdown-1", "value"),
    ],
)
def display_time_series(ticker, radioo, dpd):
    # print(type(symbol[dpd]))
    df = pd.read_csv(
        "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/raw/stock/yahoo_finance/"
        + symbol[dpd]
    )
    # replace with your own data source
    fig = px.line(df, x="Date", y=radioo)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
