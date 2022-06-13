from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from joblib import load
import pandas as pd
import dash_daq as daq

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

# loaded_rf_model = load("..\..\models\rf_models\TCS.joblib")
# random_forest_data = data_preprocessing("TCS")
# random_forest_object = RandomForest(random_forest_data)

# # Output will be -1.0 or
# random_forest_predicted = random_forest_object.predict()[0]

random_forest_predicted = 1.0

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.H1(children="Stock Market Prediction"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(children="Price Prediction"),
                        dcc.Dropdown(
                            symbol.index,
                            id="pandas-dropdown-1",
                            value=dff["Company Name"][0],
                        ),
                        html.Div(id="output-container-1"),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H2("Data analysis"),
                        dcc.Graph(id="time-series-chart"),
                        dbc.Label("Stock features"),
                        dcc.Dropdown(
                            id="stock_features",
                            options=["High", "Low", "Open", "Close"],
                            value="High",
                        ),
                    ],
                    md=6,
                ),
            ]
            style={}
        ),
    ]
)


@app.callback(
    Output("output-container-1", "children"),
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
    return container1


@app.callback(
    Output("time-series-chart", "figure"),
    [
        Input("stock_features", "value"),
        Input("pandas-dropdown-1", "value"),
    ],
)
def display_time_series(radioo, dpd):
    # print(type(symbol[dpd]))
    df = pd.read_csv(
        "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/raw/stock/yahoo_finance/"
        + symbol[dpd]
    )
    # replace with your own data source
    fig = px.line(df, x="Date", y=radioo)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
