from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from joblib import load
import pandas as pd
import dash_daq as daq

nifty_url = "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/ind_nifty50list.csv"
adani_url = "https://raw.githubusercontent.com/sudharshanavp/stock_market_prediction/machine_learning/data/raw/stock/yahoo_finance/ADANIPORTS"
nifty_df = pd.read_csv(nifty_url)
stock_df = pd.read_csv(adani_url)
symbol = pd.Series(nifty_df["Symbol"])
symbol.index = pd.Series(nifty_df["Company Name"])
symbol.to_dict()

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

pricePredictionLayout = (
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
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(id="output-container-2"),
                            ]
                        ),
                        style={"width": "100%"},
                    ),
                ],
                md=6,
                style={"padding": "1rem"},
            ),
            dbc.Col(
                [
                    html.H2("Data analysis", style={"padding-left": "20px"}),
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
            ),
        ],
        style={},
    ),
)

sentimentLayout = html.Div(
    [
        html.H2(children="Sentimental Analysis"),
        html.Label("Overall Market Sentiment: Positive"),
        html.Br(),
        html.Label("Stock sentiment: Negative"),
        html.Br(),
        html.Label("Sentimental: -0.93"),
        html.Br(),
        html.Label("Accuracy of Sentiment Analysis: 95.6%"),
    ]
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
    container1 = html.Div(
        [
            html.H2("Regression"),
            html.P("Description"),
            html.Label("Prediction Model: "),
            " LSTM 30 Day Moving Averrage",
            html.Br(),
            html.Label("Estimated Price:", style={"font-weight": "bold"}),
            dpd,
            html.Br(),
            html.Label("Mean absolute Error of Model: "),
            " 89.93",
            html.Br(),
            html.Label("Stock Trend: "),
            " Downward",
        ]
    )
    container2 = html.Div(
        [
            html.H2("Classification"),
            html.P("Description"),
            html.Label("Stock Classification: "),
            " LSTM 30 Day Moving Averrage",
            html.Br(),
            html.Label("Accuracy of Model: "),
            " 89.93",
        ]
    )
    return fig, container1, container2


if __name__ == "__main__":
    app.run_server(debug=True)
