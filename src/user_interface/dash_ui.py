from dash import Dash, dash_table, dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, precision_recall_curve

import models.random_forest_model as rf
import models.lstm_model as lstm
from joblib import load
import pandas as pd
import dash_daq as daq

cwd = Path.cwd()
root_folder = str(cwd.parent.parent).replace("\\", "/")
data_folder = str(cwd.parent.parent).replace("\\", "/") + "/data"

nse_folder = data_folder + "/processed/stocks/nse_scraped/"
yf_folder = data_folder + "/raw/stock/yahoo_finance/"

model_folder = root_folder + "/models"
random_forest_folder = model_folder + "/rf_models/"
lstm_folder = model_folder + "/lstm_models/"

nifty_path = data_folder + "/ind_nifty50list.csv"
stock_path = data_folder + "/processed/stocks/nse_scraped/ADANIPORTS.csv"
news_path = data_folder + "/dummy_news_data.csv"
nifty_df = pd.read_csv(nifty_path)
stock_df = pd.read_csv(stock_path)
news_df = pd.read_csv(news_path)
symbol = pd.Series(nifty_df["Symbol"])
symbol.index = pd.Series(nifty_df["Company Name"])
symbol.to_dict()

theme = {
    "dark": False,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}


# loaded_rf_model = load("../../models/rf_models/.joblib")

# random_forest_object = rf.RandomForest(random_forest_data)

# # Output will be -1.0 or
# random_forest_predicted = random_forest_object.predict()[0]


# random_forest_predicted = 1.0

pricePredictionLayout = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H2(
                        children="Price Prediction", style={"padding-left": "20px"}
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                symbol.index,
                                id="pandas-dropdown-1",
                                value=nifty_df["Company Name"][0],
                            ),
                            dbc.Button("Predict", id="predict_button", color="primary"),
                        ],
                        className="d-grid gap-2",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(id="output-container-1"),
                            ]
                        ),
                        style={"width": "100%"},
                        className="container1",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(id="output-container-2"),
                            ]
                        ),
                        style={"width": "100%"},
                        className="container2",
                    ),
                ],
                md=6,
                style={"padding": "1rem"},
            ),
            dbc.Col(
                [
                    html.H2(
                        "Data analysis",
                        style={"padding-left": "20px", "text-align": "center"},
                    ),
                    html.P(
                        id="company-title",
                        style={
                            "font-weight": "bold",
                            "color": "green",
                            "text-align": "center",
                            "font-size": "1.25em",
                        },
                    ),
                    dcc.Graph(id="time-series-chart"),
                    dbc.Label("Stock features"),
                    dcc.Dropdown(
                        id="stock_features",
                        options=["High", "Low", "Open", "Close"],
                        value="High",
                    ),
                ],
                md=6,
                style={"padding": "1rem"},
                className="analysis",
            ),
        ],
        style={},
    ),
]

sentimentLayout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H2(children="Sentimental Analysis"),
                    html.Label("Overall Market Sentiment: Positive"),
                    html.Br(),
                    html.Label("Stock sentiment: Negative"),
                    html.Br(),
                    html.Label("Sentimental: -0.93"),
                    html.Br(),
                    html.Label("Accuracy of Sentiment Analysis: 95.6%"),
                ],
                className="sentimental-container",
            ),
            dbc.Col(
                [
                    html.H1("Top Headlines", style={"text-align": "center"}),
                    dash_table.DataTable(
                        news_df.head(10).to_dict("records"),
                        [{"name": i, "id": i} for i in news_df.columns],
                        style_cell={"textAlign": "left"},
                    ),
                ]
            ),
        ]
    )
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Price Prediction", href="/", active="exact")),
                dbc.NavItem(
                    dbc.NavLink(
                        "Sentiment analysis",
                        href="sentimental_analysis",
                        active="exact",
                    )
                ),
            ],
            brand="Stock Market Prediction",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        html.Div(children=pricePredictionLayout, id="content"),
    ],
    fluid=True,
)


@app.callback(Output("content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return pricePredictionLayout
    elif pathname == "/sentimental_analysis":
        return sentimentLayout

    return dbc.Jumbotron(
        [
            html.H1("404: Not Found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    [
        Output("time-series-chart", "figure"),
        Output("output-container-1", "children"),
        Output("output-container-2", "children"),
        Output("company-title", "children"),
    ],
    Input("predict_button", "n_clicks"),
    [
        State("stock_features", "value"),
        State("pandas-dropdown-1", "value"),
    ],
)
def display_time_series(n, radioo, dpd):
    # print(type(symbol[dpd]))
    df = pd.read_csv(
        "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/raw/stock/yahoo_finance/"
        + symbol[dpd]
    )
    # replace with your own data source
    fig = px.line(df, x="Date", y=radioo)
    fig.update_xaxes(
        rangeslider_visible=True,
        title_text=dpd,
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

    rf_object = rf.RandomForest(nse_folder + symbol[dpd] + ".csv", 14)
    rf_object.feature_engineering()
    rf_model = load(random_forest_folder + symbol[dpd] + ".joblib")
    accuracy, _ = rf_object.test_model(rf_model)
    predicted_value = rf_object.predict_result(rf_model)[-1]

    if predicted_value > 0:
        predicted_value = "Up Day"
    else:
        predicted_value = "Down Day"

    stock_path = yf_folder + symbol[dpd]
    lstm_object = lstm.LongShortTermMemory(stock_path)
    estimated_price = float(lstm_object.predict_values(lstm_folder + symbol[dpd]))

    predicted_value  # = "DUMMY"
    accuracy  # = 74.3213213213

    estimated_price  # = 123.213
    mea = 4.5e-6

    container1 = html.Div(
        [
            html.H2("Regression"),
            html.P("Description"),
            html.Label("Prediction Model: "),
            " LSTM 10 ",
            html.Br(),
            html.Label("Estimated Price:", style={"font-weight": "bold"}),
            estimated_price,
            html.Br(),
            html.Label("Mean absolute Error of Model: "),
            mea,
        ],
        className="regression",
    )
    container2 = html.Div(
        [
            html.H2("Classification"),
            html.P("Description"),
            html.Label("Stock Classification: "),
            predicted_value,
            html.Br(),
            html.Label("Accuracy of Model (%): "),
            str(accuracy),
        ],
        className="classification",
    )
    return fig, container1, container2, dpd


if __name__ == "__main__":
    app.run_server(debug=True)
