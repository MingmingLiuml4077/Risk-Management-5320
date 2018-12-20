from __future__ import division

import dash
#from server import server
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pandas_datareader.data as web
from pandas_datareader.data import Options
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import scipy.stats as stat
import dateutil.relativedelta
import sys
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Legend
import operator
import math
import matplotlib.pyplot as plt
import datetime
from flask import Flask

print(dcc.__version__) # 0.6.0 or above is required

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True #avoid warning

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1(children="Risk Calculation System", style={'textAlign': 'center',}),
    html.H6(children="The system can show the price,VaR,ES and back test result for individual stock, portfolio, portfolio with options. You can access the system through links below.",
                    style={
                        'textAlign': 'center'
                    }
                ),
    dcc.Link('Go to Individual Stock', href='/page-1',style={'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    dcc.Link('Go to Portfolio', href='/page-2',style={'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    dcc.Link('Go to Portfolio With Option', href='/page-3',style={'textAlign': 'center'}),
])

app_1_layout = html.Div([
    # dcc.Tabs(id="tabs", value='tab-1', children=[
    #     dcc.Tab(label='Tab one', value='tab-1',children=[
         html.Div([
                html.H3(children='Individual Stock', style={
                    'textAlign': 'center',
                }),
                html.H6(
                    children='For the price and risk plots, enter the parameter below. Once a plot is generated.',
                    style={
                        'textAlign': 'center'
                    }
                ),
                html.Div([
                    html.Label('Ticker:'),
                    dcc.Input(id='ticker', type='text', value='AAPL'),

                    html.Label('Initial Investments:'),
                    dcc.Input(id='invest', type='number', value='10000'),

                    html.Label('Window (years):'),
                    dcc.Input(id='window', type='number', value='5'),

                    html.Label('Position:'),
                    dcc.Dropdown(id='position',
                                 options=[
                                     {'label': 'Long', 'value': 'long'},
                                     {'label': 'Short', 'value': 'short'}
                                 ],
                                 value='long'
                                 ,style={'width': '75%'}),
                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Start date:'),
                    dcc.Input(id="start", type='date', value='1998-01-01'),

                    html.Label('VaR Probability:'),
                    dcc.Input(id="var", type='number', value='0.99'),

                    html.Label('Horizon (days):'),
                    dcc.Input(id='horizon', type='number', value='5'),
                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('End Date'),
                    dcc.Input(id='end', type='date', value='2018-01-01'),

                    html.Label('ES Probability:'),
                    dcc.Input(id='es', type='number', value='0.975'),

                    html.Label('Risk Method:'),
                    dcc.Dropdown(id='method',
                                 options=[
                                     {'label': 'Parametric VaR/ES', 'value': 'Parametric'},
                                     {'label': 'Histroical VaR/ES', 'value': 'Historical'},
                                     {'label': 'Monte Carlo VaR/ES', 'value': 'Monte Carlo'}
                                 ],
                                 value='Parametric'
                                 ),
                ],
                    style={'width': '30%', 'display': 'inline-block','vertical-align': 'text-bottom'}),

                html.Div([
                    html.Label("Output:"),
                    dcc.Dropdown(id="output",
                                 options=[
                                     {'label': 'Price Plot', 'value': 'price'},
                                     {'label': 'Risk Plot', 'value': 'risk'},
                                     {'label': 'Backtesting Plot', 'value': "backtest"}
                                 ],
                                 value='price'
                                 ),
                    html.Br(),
                    html.Button(id='submit', n_clicks=0, children='Submit'),
                ], style={'textAlign': 'center'}),

                # html.Div([
                #     dcc.Graph(id='indicator-graphic')
                # ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})

                dcc.Graph(id='indicator-graphic-1'),]),
                #html.Div(id='indicator-graphic-1'),]),
                html.Br(),
                dcc.Link('Go to  Portfolio', href='/page-2'),
                html.Br(),
                dcc.Link('Go to Portfolio With Option', href='/page-3'),
                html.Br(),
                dcc.Link('Go back to home', href='/'),
                ])


@app.callback(
    Output('indicator-graphic-1', 'figure'),
    #Output('indicator-graphic-1', 'children'),
    [Input('submit', 'n_clicks')],
    [State("output","value"),
     State('ticker', 'value'),
     State('invest', 'value'),
     State('start', 'value'),
     State('end', 'value'),
     State("position","value"),
     State('window',"value"),
     State('var', 'value'),
     State('es', 'value'),
     State('horizon', 'value'),
     State('method', 'value')
     ])

def update_graph(nclicks,output,stock, invest, start, end,position,window,VaR_p,ES_p,horizon,method):
    if (nclicks > 0):

        pricedata = web.DataReader(stock, 'yahoo', start, end)['Adj Close'].sort_index(ascending=True)

        if (output == "price"):
            if (position == "long"):
                pos = float(1)
            else:
                pos = float(-1)
            pricedata = pd.DataFrame(pricedata)  # .reset_index()
            pricedata.columns = ["Price"]
            invest = float(invest) * pos
            amount = (invest / pricedata.iat[0, 0])
            pricedata["Price"] = pricedata["Price"] * amount

            #return u"{}".format(type(pricedata.index))
            return {
                'data': [go.Scatter(
                x=pricedata.index,
                y=pricedata['Price'],
                #text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                mode='lines+markers',
                marker={
                    'size': 1,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
                'layout': go.Layout(
                    xaxis={
                        'title': "Date",
                    },
                    yaxis={
                        'title': "Stock Price",
                    },
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                    hovermode='closest'
                )
            }

        elif (output == "risk"):
            window = int(window)
            horizon = float(horizon) / 252
            invest = float(invest)
            VaR_p = float(VaR_p)
            ES_p=float(ES_p)
            if (method == "Parametric"):
                logreturn = np.log(pricedata / pricedata.shift(1))
                logreturn = pd.DataFrame(logreturn)
                logreturn.index = pd.to_datetime(logreturn.index)
                logreturn.index = pricedata.index
                vol = logreturn.rolling(window=window * 252).std() / np.sqrt(horizon)
                mu = logreturn.rolling(window=window * 252).mean() / horizon + (vol ** 2) / 2
                if (position=="long"):
                    long_VaR = invest - invest * np.exp(vol * np.sqrt(horizon) * stat.norm.ppf(1 - VaR_p) + (mu - pow(vol, 2) / 2) * (horizon))
                    long_VaR.columns = ["VaR"]
                    #long_VaR = pd.Series(long_VaR)
                    long_ES = invest * (1 - stat.norm.cdf(stat.norm.ppf(1 - ES_p) - np.sqrt(horizon) * vol) * np.exp(mu * horizon) / (1 - ES_p))
                    long_ES.columns = ["ES"]

            #return u"{}".format(type(a.index))
                    return   {    'data': [go.Scatter(
                            x=long_VaR.index,
                            y=long_VaR["VaR"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="VaR",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        ),
                        go.Scatter(
                            x=long_ES.index,
                            y=long_ES["ES"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="ES",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        )
                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Risk Value",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif(position=="short"):
                    short_VaR = -(invest - invest * np.exp(vol * (horizon ** (0.5)) * stat.norm.ppf(VaR_p) + (mu - pow(vol, 2) / 2) * horizon))
                    short_VaR.columns = ["VaR"]
                    short_ES = (ES_p*invest * (1 - np.exp(mu * horizon) / (ES_p) * stat.norm.cdf(stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) -  invest* (1 - np.exp(mu * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1-ES_p)
                    short_ES.columns = ["ES"]
                    # window = int(window)
                    # horizon = float(horizon) / 252
                    # invest = float(invest)

        # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=short_VaR.index,
                        y=short_VaR["VaR"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                        go.Scatter(
                            x=short_ES.index,
                            y=short_ES["ES"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name='ES',
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        )
                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Risk Value",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

            if (method == "Historical"):
                logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                his_VaR = []
                his_ES = []
                his_VaR_short = []
                his_ES_short = []
                if (position == "long"):
                    for i in range(len(pricedata) - 252 * window):
                        his_VaR_list = sorted(logreturn[i:i + 252 * window])
                        #short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                        his_VaR_year = invest - invest * np.exp(his_VaR_list[12])
                        #his_VaR_year_short = ( - v0 * np.exp(short_his_VaR_list[12]))
                        his_ES_list = invest - invest * np.exp(his_VaR_list[0:31])
                       # his_ES_list_short = (v0 - v0 * np.exp(short_his_VaR_list[0:31]))
                        his_ES_years = np.mean(his_ES_list)
                        #his_ES_years_short = np.mean(his_ES_list_short)
                        his_ES.append(his_ES_years)
                        his_VaR.append(his_VaR_year)
                        #his_VaR_short.append(his_VaR_year_short)
                        #his_ES_short.append(his_ES_years_short)
                    his_result = pd.concat([pd.DataFrame(his_VaR), pd.DataFrame(his_ES)], axis=1)
                    his_result.columns = ['long_var', 'long_es']
                    # long_VaR = pd.Series(long_VaR)

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=his_result.index,
                        y=his_result["long_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=his_result.index,
                                    y=his_result["long_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif (position == "short"):
                    for i in range(len(pricedata) - 252 * window):
                        #his_VaR_list = sorted(logreturn[i:i + 252 * window])
                        short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                        #his_VaR_year = v0 - v0 * np.exp(his_VaR_list[12])
                        his_VaR_year_short = (invest - invest * np.exp(short_his_VaR_list[12]))
                        #his_ES_list = v0 - v0 * np.exp(his_VaR_list[0:31])
                        his_ES_list_short = (invest - invest * np.exp(short_his_VaR_list[0:31]))
                        #his_ES_years = np.mean(his_ES_list)
                        his_ES_years_short = np.mean(his_ES_list_short)
                        #his_ES.append(his_ES_years)
                        #his_VaR.append(his_VaR_year)
                        his_VaR_short.append(his_VaR_year_short)
                        his_ES_short.append(his_ES_years_short)
                    his_result = pd.concat([pd.DataFrame(his_VaR_short), pd.DataFrame(his_ES_short)], axis=1)
                    his_result.columns = ['short_var', 'short_es']
                    # long_VaR = pd.Series(long_VaR)

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=his_result.index,
                        y=his_result["short_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=his_result.index,
                                    y=his_result["short_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

            if (method == "Monte Carlo"):
                logreturn = np.log(pricedata / pricedata.shift(int(horizon*252)))
                MC_VaR = []
                MC_ES = []
                MC_VaR_short = []
                MC_ES_short = []
                if (position == "long"):
                    for i in range(len(pricedata) - 252 * window):
                        vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                        mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                        random = vol * stat.norm.ppf(np.random.rand())
                        MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (
                                    mu + random - pow(vol, 2) / 2) * horizon)
                        #MC_VaR_val_short = -(v0 - v0 * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_prob) + (mu + random - pow(vol, 2) / 2) * horizon))
                        #MC_ES_val_short = (0.975 * v0 * (1 - np.exp((mu + random) * horizon) / (1 - 0.025) * stat.norm.ppf(stat.norm.ppf(1 - 0.025) - horizon ** (0.5) * vol)) - v0 * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (0.025)
                        MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                        MC_VaR.append(MC_VaR_val)
                        MC_ES.append(MC_ES_val)
                        #MC_VaR_short.append(MC_VaR_val)
                        #MC_ES_short.append(MC_ES_val)
                    MC_result = pd.concat([pd.DataFrame(MC_VaR), pd.DataFrame(MC_ES)],axis=1)
                    MC_result.columns = ['long_var', 'long_es']

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=MC_result.index,
                        y=MC_result["long_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=MC_result.index,
                                    y=MC_result["long_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif (position == "short"):
                    for i in range(len(pricedata) - 252 * window):
                        vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                        mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                        random = vol * stat.norm.ppf(np.random.rand())
                        #MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon)
                        MC_VaR_val_short = -(invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon))
                        MC_ES_val_short = (ES_p * invest * (1 - np.exp((mu + random) * horizon) / (ES_p) * stat.norm.ppf(stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1-ES_p)
                        #MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                        #MC_VaR.append(MC_VaR_val)
                        #MC_ES.append(MC_ES_val)
                        MC_VaR_short.append(MC_VaR_val_short)
                        MC_ES_short.append(MC_ES_val_short)
                    MC_result = pd.concat([pd.DataFrame(MC_VaR_short), pd.DataFrame(MC_ES_short)], axis=1)
                    MC_result.columns = ['short_var', 'short_es']

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=MC_result.index,
                        y=MC_result["short_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                        go.Scatter(
                            x=MC_result.index,
                            y=MC_result["short_es"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="ES",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }),

                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

        elif (output == "backtest"):
             def calc_lost(v0, horizon, price):
                 share_change = np.divide(price[:(len(price) - int(horizon * 252))], price[int(horizon * 252):])
                 loss = v0 - share_change * v0
                 return loss

             def num_exp(v0, price, var):
                 numberexception = []
                 for i in range(len(var)):
                     window = price[(len(var) - i - 252):(len(var) - i - 1)]
                     exception = 0
                     for j in range(len(window) - 4):
                         share = v0 / window[j]
                         price0 = window[j]
                         pricet = window[j + 4]
                         loss = v0 - pricet * share
                         if loss > var[len(var) - i - 252 + j]:
                             exception = exception + 1
                         else:
                             exception = exception

                     numberexception.append(exception)
                 return numberexception

             window = int(window)
             horizon = float(horizon) / 252
             invest = float(invest)
             VaR_p = float(VaR_p)
             ES_p = float(ES_p)
             if (method == "Parametric"):
                 logreturn = np.log(pricedata / pricedata.shift(1))
                 logreturn = pd.DataFrame(logreturn)
                 logreturn.index = pd.to_datetime(logreturn.index)
                 logreturn.index = pricedata.index
                 vol = logreturn.rolling(window=window * 252).std() / np.sqrt(horizon)
                 mu = logreturn.rolling(window=window * 252).mean() / horizon + (vol ** 2) / 2
                 if (position == "long"):
                     long_VaR = invest - invest * np.exp(
                         vol * np.sqrt(horizon) * stat.norm.ppf(1 - VaR_p) + (mu - pow(vol, 2) / 2) * (horizon))
                     long_VaR.columns = ["VaR"]
                     # long_VaR = pd.Series(long_VaR)
                     long_ES = invest * (1 - stat.norm.cdf(stat.norm.ppf(1 - ES_p) - np.sqrt(horizon) * vol) * np.exp(
                         mu * horizon) / (1 - ES_p))
                     long_ES.columns = ["ES"]
                     result= pd.concat([pd.DataFrame(long_VaR), pd.DataFrame(long_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']


                 elif (position == "short"):
                     short_VaR = -(invest - invest * np.exp(
                         vol * (horizon ** (0.5)) * stat.norm.ppf(VaR_p) + (mu - pow(vol, 2) / 2) * horizon))
                     #short_VaR.columns = ["VaR"]
                     short_ES = (ES_p * invest * (1 - np.exp(mu * horizon) / (ES_p) * stat.norm.cdf(
                         stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (
                                             1 - np.exp(mu * horizon) / (1) * stat.norm.cdf(
                                         stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1 - ES_p)
                     #short_ES.columns = ["ES"]
                     result = pd.concat([pd.DataFrame(short_VaR), pd.DataFrame(short_ES)], axis=1)
                     result.columns = ['short_var', 'short_es']


             elif (method == "Historical"):
                 logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                 his_VaR = []
                 his_ES = []
                 his_VaR_short = []
                 his_ES_short = []
                 if (position == "long"):
                     for i in range(len(pricedata) - 252 * window):
                         his_VaR_list = sorted(logreturn[i:i + 252 * window])
                         # short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                         his_VaR_year = invest - invest * np.exp(his_VaR_list[12])
                         # his_VaR_year_short = ( - v0 * np.exp(short_his_VaR_list[12]))
                         his_ES_list = invest - invest * np.exp(his_VaR_list[0:31])
                         # his_ES_list_short = (v0 - v0 * np.exp(short_his_VaR_list[0:31]))
                         his_ES_years = np.mean(his_ES_list)
                         # his_ES_years_short = np.mean(his_ES_list_short)
                         his_ES.append(his_ES_years)
                         his_VaR.append(his_VaR_year)
                         # his_VaR_short.append(his_VaR_year_short)
                         # his_ES_short.append(his_ES_years_short)
                     result = pd.concat([pd.DataFrame(his_VaR), pd.DataFrame(his_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']
                     # long_VaR = pd.Series(long_VaR)

                     # return u"{}".format(type(a.index))

                 elif (position == "short"):
                     for i in range(len(pricedata) - 252 * window):
                         # his_VaR_list = sorted(logreturn[i:i + 252 * window])
                         short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                         # his_VaR_year = v0 - v0 * np.exp(his_VaR_list[12])
                         his_VaR_year_short = (invest - invest * np.exp(short_his_VaR_list[12]))
                         # his_ES_list = v0 - v0 * np.exp(his_VaR_list[0:31])
                         his_ES_list_short = (invest - invest * np.exp(short_his_VaR_list[0:31]))
                         # his_ES_years = np.mean(his_ES_list)
                         his_ES_years_short = np.mean(his_ES_list_short)
                         # his_ES.append(his_ES_years)
                         # his_VaR.append(his_VaR_year)
                         his_VaR_short.append(his_VaR_year_short)
                         his_ES_short.append(his_ES_years_short)
                     result = pd.concat([pd.DataFrame(his_VaR_short), pd.DataFrame(his_ES_short)], axis=1)
                     result.columns = ['short_var', 'short_es']
                     # long_VaR = pd.Series(long_VaR)

                     # return u"{}".format(type(a.index))


             if (method == "Monte Carlo"):
                 logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                 MC_VaR = []
                 MC_ES = []
                 MC_VaR_short = []
                 MC_ES_short = []
                 if (position == "long"):
                     for i in range(len(pricedata) - 252 * window):
                         vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                         mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                         random = vol * stat.norm.ppf(np.random.rand())
                         MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (
                                 mu + random - pow(vol, 2) / 2) * horizon)
                         # MC_VaR_val_short = -(v0 - v0 * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_prob) + (mu + random - pow(vol, 2) / 2) * horizon))
                         # MC_ES_val_short = (0.975 * v0 * (1 - np.exp((mu + random) * horizon) / (1 - 0.025) * stat.norm.ppf(stat.norm.ppf(1 - 0.025) - horizon ** (0.5) * vol)) - v0 * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (0.025)
                         MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(
                             stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                         MC_VaR.append(MC_VaR_val)
                         MC_ES.append(MC_ES_val)
                         # MC_VaR_short.append(MC_VaR_val)
                         # MC_ES_short.append(MC_ES_val)
                     result = pd.concat([pd.DataFrame(MC_VaR), pd.DataFrame(MC_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']

                     # return u"{}".format(type(a.index))

                 elif (position == "short"):
                     for i in range(len(pricedata) - 252 * window):
                         vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                         mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                         random = vol * stat.norm.ppf(np.random.rand())
                         # MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon)
                         MC_VaR_val_short = -(invest - invest * np.exp(
                             vol * horizon ** (0.5) * stat.norm.ppf(VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon))
                         MC_ES_val_short = (ES_p * invest * (
                                     1 - np.exp((mu + random) * horizon) / (ES_p) * stat.norm.ppf(
                                 stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (
                                                        1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(
                                                    stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1 - ES_p)
                         # MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                         # MC_VaR.append(MC_VaR_val)
                         # MC_ES.append(MC_ES_val)
                         MC_VaR_short.append(MC_VaR_val_short)
                         MC_ES_short.append(MC_ES_val_short)
                     result = pd.concat([pd.DataFrame(MC_VaR_short), pd.DataFrame(MC_ES_short)], axis=1)
                     result.columns = ['short_var', 'short_es']
             loss = calc_lost(invest, horizon, pricedata)
             num = num_exp(invest, pricedata, result.iloc[0])

             return {
                'data': [
                    go.Scatter(
                        x=result.index,
                        y=result.iloc[0],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name = "VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                    go.Scatter(
                        x=loss.index,
                        y=pd.Series(loss.iloc[0]),
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="Actual Loss",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                    go.Scatter(
                        x=loss.index,
                        y=pd.Series(num),
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="num",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': "Date",
                    },
                    yaxis={
                        'title': "Stock Price",
                    },
                    margin={'l': 30, 'b': 40, 't': 40, 'r': 0},
                    hovermode='closest'
                )
            }






app_2_layout = html.Div([
         html.Div([
                html.H3(children='Portfolio', style={
                    'textAlign': 'center',
                }),
                html.H6(
                    children='For the price and risk plots of portfolio, enter the parameter below.You need to split stocks by comma.',
                    style={
                        'textAlign': 'center'
                    }
                ),
                html.Div([
                    html.Label('Ticker:'),
                    dcc.Input(id='ticker', type='text', value='AAPL,MSFT'),
                    html.Label('Initial Investments:'),
                    dcc.Input(id='invest', type='text', value='10000'),

                    html.Label('Window (years):'),
                    dcc.Input(id='window', type='text', value='5'),
                    html.Label("Weights:"),
                    dcc.Input(id='weights', type='text', value='0.5,0.5'),
                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Start date:'),
                    dcc.Input(id="start", type='date', value='1998-01-01'),

                    html.Label('VaR Probability:'),
                    dcc.Input(id="var", type='text', value='0.99'),

                    html.Label('Horizon (days):'),
                    dcc.Input(id='horizon', type='text', value='5'),

                    html.Label('Position:'),
                    dcc.Dropdown(id='position',
                                 options=[
                                     {'label': 'Long', 'value': 'long'},
                                     {'label': 'Short', 'value': 'short'}
                                 ],
                                 value='long'
                                 , style={'width': '75%'}),
                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('End Date'),
                    dcc.Input(id='end', type='date', value='2018-01-01'),

                    html.Label('ES Probability:'),
                    dcc.Input(id='es', type='text', value='0.975'),

                    html.Label('Risk Method:'),
                    dcc.Dropdown(id='method',
                                 options=[
                                     {'label': 'Parametric VaR/ES', 'value': 'Parametric'},
                                     {'label': 'Histroical VaR/ES', 'value': 'Historical'},
                                     {'label': 'Monte Carlo VaR/ES', 'value': 'Monte Carlo'}
                                 ],
                                 value='Parametric'
                                 ),
                ],
                    style={'width': '30%', 'display': 'inline-block','vertical-align': 'text-bottom'}),
                html.Br(),
                html.Div([
                    html.Label("Output:"),
                    dcc.Dropdown(id="output",
                                 options=[
                                     {'label': 'Price Plot', 'value': 'price'},
                                     {'label': 'Risk Plot', 'value': 'risk'},
                                     {'label': 'Backtesting Plot', 'value': "backtest"}
                                 ],
                                 value='price'
                                 )]),
                html.Br(),
                html.Div([
                    html.Button(id='submit', n_clicks=0, children='Submit'),
                ], style={'textAlign': 'center'}),

                # html.Div([
                #     dcc.Graph(id='indicator-graphic')
                # ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
                dcc.Graph(id='indicator-graphic-2'),
                #html.Div(id='indicator-graphic-2'),
                html.Br(),
                dcc.Link('Go to Individual Stock', href='/page-1'),
                html.Br(),
                dcc.Link('Go to Portfolio With Option', href='/page-3'),
                html.Br(),
                dcc.Link('Go back to home', href='/'),
                ]),


                ])
@app.callback(
    #Output('indicator-graphic-2', 'children'),
    Output('indicator-graphic-2', 'figure'),
    [Input('submit', 'n_clicks')],
    [State("output", "value"),
     State('ticker', 'value'),
     State('invest', 'value'),
     State('start', 'value'),
     State('end', 'value'),
     State("weights", "value"),
     State('window', "value"),
     State('var', 'value'),
     State('es', 'value'),
     State('horizon', 'value'),
     State('method', 'value'),
     State("position","value")
     ])

def update_graph2(nclicks,output,stock,invest,start,end,weights,window,VaR_p,ES_p,horizon,method,position):
    if (nclicks > 0):
        weight = weights.split(",")
        weight = list(map(float, weight))
        # return u"{}{} {}".format(float(weight[0]),float(weight[1]),float(weight[2]))
        stock = stock.split(",")
        pricedata = web.DataReader(stock, 'yahoo', start, end)['Adj Close'].sort_index(ascending=True)
        pricedata = pd.DataFrame(pricedata)  # .reset_index()
        # pricedata.columns = ["Price"]
        invest = float(invest)
        amount = np.array(weight) * invest
        priceall = amount * pricedata
        priceall = priceall.sum(axis=1)
        if (output == "price"):
            # weight = weights.split(",")
            # weight = list(map(float,weight))
            # #return u"{}{} {}".format(float(weight[0]),float(weight[1]),float(weight[2]))
            # stock =stock.split(",")
            # pricedata = web.DataReader(stock, 'yahoo', start, end)['Adj Close'].sort_index(ascending=True)
            # pricedata = pd.DataFrame(pricedata)#.reset_index()
            # #pricedata.columns = ["Price"]
            # invest = float(invest)
            # amount = np.array(weight)*invest
            # priceall = amount*pricedata
            #priceall= priceall.sum(axis=1)

            return {
                'data': [
                    go.Scatter(
                    x=pricedata.index,
                    y=priceall,
                    #text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                    mode='lines+markers',
                    marker={
                        'size': 1,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                    }

            )
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': "Date",
                    },
                    yaxis={
                        'title': "Stock Price",
                    },
                    margin={'l': 30, 'b': 40, 't': 40, 'r': 0},
                    hovermode='closest'
                )
            }

        elif (output == "risk"):
            window = int(window)
            horizon = float(horizon) / 252
            invest = float(invest)
            pricedata = priceall
            VaR_p = float(VaR_p)
            ES_p=float(ES_p)
            if (method == "Parametric"):
                logreturn = np.log(pricedata / pricedata.shift(1))
                logreturn = pd.DataFrame(logreturn)
                logreturn.index = pd.to_datetime(logreturn.index)
                logreturn.index = pricedata.index
                vol = logreturn.rolling(window=window * 252).std() / np.sqrt(horizon)
                mu = logreturn.rolling(window=window * 252).mean() / horizon + (vol ** 2) / 2
                if (position=="long"):
                    long_VaR = invest - invest * np.exp(vol * np.sqrt(horizon) * stat.norm.ppf(1 - VaR_p) + (mu - pow(vol, 2) / 2) * (horizon))
                    long_VaR.columns = ["VaR"]
                    #long_VaR = pd.Series(long_VaR)
                    long_ES = invest * (1 - stat.norm.cdf(stat.norm.ppf(1 - ES_p) - np.sqrt(horizon) * vol) * np.exp(mu * horizon) / (1 - ES_p))
                    long_ES.columns = ["ES"]

            #return u"{}".format(type(a.index))
                    return   {    'data': [go.Scatter(
                            x=long_VaR.index,
                            y=long_VaR["VaR"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="VaR",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        ),
                        go.Scatter(
                            x=long_ES.index,
                            y=long_ES["ES"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="ES",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        )
                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Risk Value",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif(position=="short"):
                    short_VaR = -(invest - invest * np.exp(vol * (horizon ** (0.5)) * stat.norm.ppf(VaR_p) + (mu - pow(vol, 2) / 2) * horizon))
                    short_VaR.columns = ["VaR"]
                    short_ES = (ES_p*invest * (1 - np.exp(mu * horizon) / (ES_p) * stat.norm.cdf(stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) -  invest* (1 - np.exp(mu * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1-ES_p)
                    short_ES.columns = ["ES"]
                    # window = int(window)
                    # horizon = float(horizon) / 252
                    # invest = float(invest)

        # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=short_VaR.index,
                        y=short_VaR["VaR"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                        go.Scatter(
                            x=short_ES.index,
                            y=short_ES["ES"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name='ES',
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }
                        )
                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Risk Value",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

            if (method == "Historical"):
                logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                his_VaR = []
                his_ES = []
                his_VaR_short = []
                his_ES_short = []
                if (position == "long"):
                    for i in range(len(pricedata) - 252 * window):
                        his_VaR_list = sorted(logreturn[i:i + 252 * window])
                        #short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                        his_VaR_year = invest - invest * np.exp(his_VaR_list[12])
                        #his_VaR_year_short = ( - v0 * np.exp(short_his_VaR_list[12]))
                        his_ES_list = invest - invest * np.exp(his_VaR_list[0:31])
                       # his_ES_list_short = (v0 - v0 * np.exp(short_his_VaR_list[0:31]))
                        his_ES_years = np.mean(his_ES_list)
                        #his_ES_years_short = np.mean(his_ES_list_short)
                        his_ES.append(his_ES_years)
                        his_VaR.append(his_VaR_year)
                        #his_VaR_short.append(his_VaR_year_short)
                        #his_ES_short.append(his_ES_years_short)
                    his_result = pd.concat([pd.DataFrame(his_VaR), pd.DataFrame(his_ES)], axis=1)
                    his_result.columns = ['long_var', 'long_es']
                    # long_VaR = pd.Series(long_VaR)

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=his_result.index,
                        y=his_result["long_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=his_result.index,
                                    y=his_result["long_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif (position == "short"):
                    for i in range(len(pricedata) - 252 * window):
                        #his_VaR_list = sorted(logreturn[i:i + 252 * window])
                        short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                        #his_VaR_year = v0 - v0 * np.exp(his_VaR_list[12])
                        his_VaR_year_short = (invest - invest * np.exp(short_his_VaR_list[12]))
                        #his_ES_list = v0 - v0 * np.exp(his_VaR_list[0:31])
                        his_ES_list_short = (invest - invest * np.exp(short_his_VaR_list[0:31]))
                        #his_ES_years = np.mean(his_ES_list)
                        his_ES_years_short = np.mean(his_ES_list_short)
                        #his_ES.append(his_ES_years)
                        #his_VaR.append(his_VaR_year)
                        his_VaR_short.append(his_VaR_year_short)
                        his_ES_short.append(his_ES_years_short)
                    his_result = pd.concat([pd.DataFrame(his_VaR_short), pd.DataFrame(his_ES_short)], axis=1)
                    his_result.columns = ['short_var', 'short_es']
                    # long_VaR = pd.Series(long_VaR)

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=his_result.index,
                        y=his_result["short_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=his_result.index,
                                    y=his_result["short_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

            if (method == "Monte Carlo"):
                logreturn = np.log(pricedata / pricedata.shift(int(horizon*252)))
                MC_VaR = []
                MC_ES = []
                MC_VaR_short = []
                MC_ES_short = []
                if (position == "long"):
                    for i in range(len(pricedata) - 252 * window):
                        vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                        mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                        random = vol * stat.norm.ppf(np.random.rand())
                        MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (
                                    mu + random - pow(vol, 2) / 2) * horizon)
                        #MC_VaR_val_short = -(v0 - v0 * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_prob) + (mu + random - pow(vol, 2) / 2) * horizon))
                        #MC_ES_val_short = (0.975 * v0 * (1 - np.exp((mu + random) * horizon) / (1 - 0.025) * stat.norm.ppf(stat.norm.ppf(1 - 0.025) - horizon ** (0.5) * vol)) - v0 * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (0.025)
                        MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                        MC_VaR.append(MC_VaR_val)
                        MC_ES.append(MC_ES_val)
                        #MC_VaR_short.append(MC_VaR_val)
                        #MC_ES_short.append(MC_ES_val)
                    MC_result = pd.concat([pd.DataFrame(MC_VaR), pd.DataFrame(MC_ES)],axis=1)
                    MC_result.columns = ['long_var', 'long_es']

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=MC_result.index,
                        y=MC_result["long_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                         go.Scatter(
                                    x=MC_result.index,
                                    y=MC_result["long_es"],
                                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                    mode='lines+markers',
                                    name="ES",
                                    marker={
                                        'size': 1,
                                        'opacity': 0.5,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    }),

                                     ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }
                elif (position == "short"):
                    for i in range(len(pricedata) - 252 * window):
                        vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                        mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                        random = vol * stat.norm.ppf(np.random.rand())
                        #MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon)
                        MC_VaR_val_short = -(invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon))
                        MC_ES_val_short = (ES_p * invest * (1 - np.exp((mu + random) * horizon) / (ES_p) * stat.norm.ppf(stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1-ES_p)
                        #MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                        #MC_VaR.append(MC_VaR_val)
                        #MC_ES.append(MC_ES_val)
                        MC_VaR_short.append(MC_VaR_val_short)
                        MC_ES_short.append(MC_ES_val_short)
                    MC_result = pd.concat([pd.DataFrame(MC_VaR_short), pd.DataFrame(MC_ES_short)], axis=1)
                    MC_result.columns = ['short_var', 'short_es']

                    # return u"{}".format(type(a.index))
                    return {'data': [go.Scatter(
                        x=MC_result.index,
                        y=MC_result["short_var"],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    ),
                        go.Scatter(
                            x=MC_result.index,
                            y=MC_result["short_es"],
                            # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                            mode='lines+markers',
                            name="ES",
                            marker={
                                'size': 1,
                                'opacity': 0.5,
                                'line': {'width': 0.5, 'color': 'white'}
                            }),

                    ],
                        'layout': go.Layout(
                            xaxis={
                                'title': "Date",
                            },
                            yaxis={
                                'title': "Stock Price",
                            },
                            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                            hovermode='closest'
                        )
                    }

        elif (output == "backtest"):
             pricedata = priceall

             def calc_lost(v0, horizon, price):
                 share_change = np.divide(price[:(len(price) - int(horizon * 252))], price[int(horizon * 252):])
                 loss = v0 - share_change * v0
                 return loss

             def num_exp(v0, price, var):
                 numberexception = []
                 for i in range(len(var)):
                     window = price[(len(var) - i - 252):(len(var) - i - 1)]
                     exception = 0
                     for j in range(len(window) - 4):
                         share = v0 / window[j]
                         price0 = window[j]
                         pricet = window[j + 4]
                         loss = v0 - pricet * share
                         if loss > var[len(var) - i - 252 + j]:
                             exception = exception + 1
                         else:
                             exception = exception

                     numberexception.append(exception)
                 return numberexception

             window = int(window)
             horizon = float(horizon) / 252
             invest = float(invest)
             VaR_p = float(VaR_p)
             ES_p = float(ES_p)
             if (method == "Parametric"):
                 logreturn = np.log(pricedata / pricedata.shift(1))
                 logreturn = pd.DataFrame(logreturn)
                 logreturn.index = pd.to_datetime(logreturn.index)
                 logreturn.index = pricedata.index
                 vol = logreturn.rolling(window=window * 252).std() / np.sqrt(horizon)
                 mu = logreturn.rolling(window=window * 252).mean() / horizon + (vol ** 2) / 2
                 if (position == "long"):
                     long_VaR = invest - invest * np.exp(
                         vol * np.sqrt(horizon) * stat.norm.ppf(1 - VaR_p) + (mu - pow(vol, 2) / 2) * (horizon))
                     long_VaR.columns = ["VaR"]
                     # long_VaR = pd.Series(long_VaR)
                     long_ES = invest * (1 - stat.norm.cdf(stat.norm.ppf(1 - ES_p) - np.sqrt(horizon) * vol) * np.exp(
                         mu * horizon) / (1 - ES_p))
                     long_ES.columns = ["ES"]
                     result= pd.concat([pd.DataFrame(long_VaR), pd.DataFrame(long_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']


                 elif (position == "short"):
                     short_VaR = -(invest - invest * np.exp(
                         vol * (horizon ** (0.5)) * stat.norm.ppf(VaR_p) + (mu - pow(vol, 2) / 2) * horizon))
                     #short_VaR.columns = ["VaR"]
                     short_ES = (ES_p * invest * (1 - np.exp(mu * horizon) / (ES_p) * stat.norm.cdf(
                         stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (
                                             1 - np.exp(mu * horizon) / (1) * stat.norm.cdf(
                                         stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1 - ES_p)
                     #short_ES.columns = ["ES"]
                     result = pd.concat([pd.DataFrame(short_VaR), pd.DataFrame(short_ES)], axis=1)
                     result.columns = ['short_var', 'short_es']


             elif (method == "Historical"):
                 logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                 his_VaR = []
                 his_ES = []
                 his_VaR_short = []
                 his_ES_short = []
                 if (position == "long"):
                     for i in range(len(pricedata) - 252 * window):
                         his_VaR_list = sorted(logreturn[i:i + 252 * window])
                         # short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                         his_VaR_year = invest - invest * np.exp(his_VaR_list[12])
                         # his_VaR_year_short = ( - v0 * np.exp(short_his_VaR_list[12]))
                         his_ES_list = invest - invest * np.exp(his_VaR_list[0:31])
                         # his_ES_list_short = (v0 - v0 * np.exp(short_his_VaR_list[0:31]))
                         his_ES_years = np.mean(his_ES_list)
                         # his_ES_years_short = np.mean(his_ES_list_short)
                         his_ES.append(his_ES_years)
                         his_VaR.append(his_VaR_year)
                         # his_VaR_short.append(his_VaR_year_short)
                         # his_ES_short.append(his_ES_years_short)
                     result = pd.concat([pd.DataFrame(his_VaR), pd.DataFrame(his_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']
                     # long_VaR = pd.Series(long_VaR)

                     # return u"{}".format(type(a.index))

                 elif (position == "short"):
                     for i in range(len(pricedata) - 252 * window):
                         # his_VaR_list = sorted(logreturn[i:i + 252 * window])
                         short_his_VaR_list = sorted(-logreturn[i:i + 252 * window])
                         # his_VaR_year = v0 - v0 * np.exp(his_VaR_list[12])
                         his_VaR_year_short = (invest - invest * np.exp(short_his_VaR_list[12]))
                         # his_ES_list = v0 - v0 * np.exp(his_VaR_list[0:31])
                         his_ES_list_short = (invest - invest * np.exp(short_his_VaR_list[0:31]))
                         # his_ES_years = np.mean(his_ES_list)
                         his_ES_years_short = np.mean(his_ES_list_short)
                         # his_ES.append(his_ES_years)
                         # his_VaR.append(his_VaR_year)
                         his_VaR_short.append(his_VaR_year_short)
                         his_ES_short.append(his_ES_years_short)
                     result = pd.concat([pd.DataFrame(his_VaR_short), pd.DataFrame(his_ES_short)], axis=1)
                     result.columns = ['short_var', 'short_es']
                     # long_VaR = pd.Series(long_VaR)

                     # return u"{}".format(type(a.index))


             if (method == "Monte Carlo"):
                 logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
                 MC_VaR = []
                 MC_ES = []
                 MC_VaR_short = []
                 MC_ES_short = []
                 if (position == "long"):
                     for i in range(len(pricedata) - 252 * window):
                         vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                         mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                         random = vol * stat.norm.ppf(np.random.rand())
                         MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (
                                 mu + random - pow(vol, 2) / 2) * horizon)
                         # MC_VaR_val_short = -(v0 - v0 * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(VaR_prob) + (mu + random - pow(vol, 2) / 2) * horizon))
                         # MC_ES_val_short = (0.975 * v0 * (1 - np.exp((mu + random) * horizon) / (1 - 0.025) * stat.norm.ppf(stat.norm.ppf(1 - 0.025) - horizon ** (0.5) * vol)) - v0 * (1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (0.025)
                         MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(
                             stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                         MC_VaR.append(MC_VaR_val)
                         MC_ES.append(MC_ES_val)
                         # MC_VaR_short.append(MC_VaR_val)
                         # MC_ES_short.append(MC_ES_val)
                     result = pd.concat([pd.DataFrame(MC_VaR), pd.DataFrame(MC_ES)], axis=1)
                     result.columns = ['long_var', 'long_es']

                     # return u"{}".format(type(a.index))

                 elif (position == "short"):
                     for i in range(len(pricedata) - 252 * window):
                         vol = np.std(logreturn[i:i + 252 * window]) / np.sqrt(horizon)
                         mu = np.mean(logreturn[i:i + 252 * window]) / horizon + (vol ** 2) / 2
                         random = vol * stat.norm.ppf(np.random.rand())
                         # MC_VaR_val = invest - invest * np.exp(vol * horizon ** (0.5) * stat.norm.ppf(1 - VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon)
                         MC_VaR_val_short = -(invest - invest * np.exp(
                             vol * horizon ** (0.5) * stat.norm.ppf(VaR_p) + (mu + random - pow(vol, 2) / 2) * horizon))
                         MC_ES_val_short = (ES_p * invest * (
                                     1 - np.exp((mu + random) * horizon) / (ES_p) * stat.norm.ppf(
                                 stat.norm.ppf(ES_p) - horizon ** (0.5) * vol)) - invest * (
                                                        1 - np.exp((mu + random) * horizon) / (1) * stat.norm.cdf(
                                                    stat.norm.ppf(1) - horizon ** (0.5) * vol))) / (1 - ES_p)
                         # MC_ES_val = invest * (1 - np.exp((mu + random) * horizon) / (1 - ES_p) * stat.norm.cdf(stat.norm.ppf(1 - ES_p) - horizon ** (0.5) * vol))
                         # MC_VaR.append(MC_VaR_val)
                         # MC_ES.append(MC_ES_val)
                         MC_VaR_short.append(MC_VaR_val_short)
                         MC_ES_short.append(MC_ES_val_short)
                     result = pd.concat([pd.DataFrame(MC_VaR_short), pd.DataFrame(MC_ES_short)], axis=1)
                     result.columns = ['short_var', 'short_es']
             loss = calc_lost(invest, horizon, pricedata)
             num = num_exp(invest, pricedata, result.iloc[0])

             return {
                'data': [
                    go.Scatter(
                        x=result.index,
                        y=result.iloc[0],
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name = "VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                    go.Scatter(
                        x=loss.index,
                        y=pd.Series(loss.iloc[0]),
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="Actual Loss",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                    go.Scatter(
                        x=loss.index,
                        y=pd.Series(num),
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="num",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }

                    ),
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': "Date",
                    },
                    yaxis={
                        'title': "Stock Price",
                    },
                    margin={'l': 30, 'b': 40, 't': 40, 'r': 0},
                    hovermode='closest'
                )
            }


app_3_layout = html.Div([
         html.Div([
                html.H3(children='Portfolio with Option', style={
                    'textAlign': 'center',
                }),
                html.H6(
                    children='This page can calculate VaR that can be reduced by liquidating a portion of a single stock portfolio to hedge with put option',
                    style={
                        'textAlign': 'center'
                    }
                ),
                html.Div([
                    html.Label('Ticker:'),
                    dcc.Input(id='ticker', type='text', value='AAPL'),

                    html.Label('Initial Investments:'),
                    dcc.Input(id='invest', type='text', value='10000'),

                    html.Label('Window (years):'),
                    dcc.Input(id='window', type='text', value='5'),

                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Start date:'),
                    dcc.Input(id="start", type='date', value='1998-01-01'),

                    html.Label('VaR Probability:'),
                    dcc.Input(id="var", type='text', value='0.99'),

                    html.Label('Horizon (days):'),
                    dcc.Input(id='horizon', type='text', value='5'),

                ],
                    style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('End Date'),
                    dcc.Input(id='end', type='date', value='2018-01-01'),

                    html.Label("Weights:"),
                    dcc.Input(id='weights', type='text', value='0.9,0.1'),

                    html.Label('Risk Method:'),
                    dcc.Dropdown(id='method',
                                 options=[
                                     {'label': 'Parametric VaR/ES', 'value': 'Parametric'},
                                     {'label': 'Histroical VaR/ES', 'value': 'Historical'},
                                     {'label': 'Monte Carlo VaR/ES', 'value': 'Monte Carlo'}
                                 ],
                                 value='Parametric'
                                 ),
                ],
                    style={'width': '30%', 'display': 'inline-block','vertical-align': 'text-bottom'}),
                html.Br(),
                html.Div([
                    html.Label("Output:"),
                    dcc.Dropdown(id="output",
                                 options=[
                                     {'label': 'Price Plot', 'value': 'price'},
                                     {'label': 'Risk Plot', 'value': 'risk'},
                                     {'label': 'Backtesting Plot', 'value': "backtest"}
                                 ],
                                 value='price'
                                 )]),
                html.Br(),
                html.Div([
                    html.Button(id='submit', n_clicks=0, children='Submit'),
                ], style={'textAlign': 'center'}),

                # html.Div([
                #     dcc.Graph(id='indicator-graphic')
                # ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
                dcc.Graph(id='indicator-graphic-3'),
                #html.Div(id='indicator-graphic-3'),
                html.Br(),
                dcc.Link('Go to Individual Stock', href='/page-1'),
                html.Br(),
                dcc.Link('Go to Portfolio With Option', href='/page-3'),
                html.Br(),
                dcc.Link('Go back to home', href='/'),
                ]),


                ])
@app.callback(
    #Output('indicator-graphic-3', 'children'),
    Output('indicator-graphic-3', 'figure'),
    [Input('submit', 'n_clicks')],
    [State("output", "value"),
     State('ticker', 'value'),
     State('invest', 'value'),
     State('start', 'value'),
     State('end', 'value'),
     State("weights", "value"),
     State('window', "value"),
     State('var', 'value'),
     State('horizon', 'value'),
     State('method', 'value'),
     ])

def update_graph3(nclicks,output,stock,invest,start,end,weights,window,VaR_p,horizon,method):
    if (nclicks > 0):
        def euro_vanilla_put(S, K, T, r, sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                put = (K * np.exp(-r * T) * stat.norm.cdf(-d2, 0.0, 1.0) - S * stat.norm.cdf(-d1, 0.0, 1.0))
                delta = -stat.norm.cdf(-d1)
                vega = stat.norm.pdf(d1) * S * np.sqrt(T)
                return put


        weight = weights.split(",")
        weight = list(map(float, weight))
        # return u"{}{} {}".format(float(weight[0]),float(weight[1]),float(weight[2]))
        stock = stock.split(",")
        pricedata = web.DataReader(stock, 'yahoo', start, end)['Adj Close'].sort_index(ascending=True)
        pricedata = pd.DataFrame(pricedata)  # .reset_index()
        T = 1
        r = 0.005
        window = int(window)
        horizon = float(horizon) / 252
        VaR_p = float(VaR_p)
        logreturn = np.log(pricedata / pricedata.shift(int(horizon * 252)))
        logreturn = pd.DataFrame(logreturn)
        logreturn.index = pd.to_datetime(logreturn.index)
        logreturn.index = pricedata.index
        vol = logreturn.rolling(window=window * 252).std() / np.sqrt(horizon)
        mu = logreturn.rolling(window=window * 252).mean() / horizon + (vol ** 2) / 2

        est_p0 = euro_vanilla_put(pricedata, pricedata, T, r, vol)

        def Option_para(S0, r, sigma, T):
            """opt_type: put or call """
            d1 = ((r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            delta = -stat.norm.cdf(-d1)
            vega = stat.norm.pdf(d1) * S0 * np.sqrt(T)
            return delta, vega

        est_delta, est_vega = Option_para(pricedata, r, vol, T)

        pricedata1 = pd.concat([pd.DataFrame(pricedata),pd.DataFrame(est_p0)],axis=1)

        # pricedata.columns = ["Price"]
        invest = float(invest)
        amount = np.array(weight) * invest
        priceall = amount * pricedata1
        priceall = priceall.sum(axis=1)

        if (output == "price"):

            return {
                'data': [go.Scatter(
                x=priceall.index,
                y=pd.Series(priceall),
                #     #text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                mode='lines+markers',
                marker={
                'size': 1,
               'opacity': 0.5,
                 'line': {'width': 0.5, 'color': 'white'}
                   }

                 )
                  ],


                       'layout': go.Layout(
                           xaxis={
                                'title': "Date",
                             },
                            yaxis={
                               'title': "Stock Price",
                          },
                           margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                           hovermode='closest'
                        )
                   }
        elif (output == "risk"):

            if (method == "Parametric"):
                def VaR_option_para(p0, S0, delta, sigma0, mu, T, invest_1, invest_2, p):
                    share1 = invest_1 / S0
                    share2 = invest_2 / p0
                    VaR = share2 * p0 + share1 * S0 + delta * share2 * S0 - share2 * p0 - (
                                share1 + delta * share2) * S0 * np.exp(
                        sigma0 * np.sqrt(T) * stat.norm.ppf(1 - p) + (mu - sigma0 ** 2 / 2) * T)
                    return (VaR)

                para_var = VaR_option_para(est_p0,pricedata, est_delta,vol,mu, T, amount[0], amount[1], VaR_p)
                para_var = para_var.dropna()

                return   {    'data': [go.Scatter(
                        x=para_var.index,
                        y=pd.Series(para_var.iloc[1260:,0]),
                        # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                        mode='lines+markers',
                        name="VaR",
                        marker={
                            'size': 1,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        }
                    )
                ],
                    'layout': go.Layout(
                        xaxis={
                            'title': "Date",
                        },
                        yaxis={
                            'title': "Risk Value",
                        },
                        margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                        hovermode='closest'
                    )
                }
            #
            # if (method == "Historical"):
            #     def his_optionloss(invest1, invest2, st, T, r, windowlen, time, opt_type="put"):
            #         a = int(time * 252)
            #         logreturn = np.log(st / st.shift(a))
            #         sigma = logreturn.rolling(window=windowlen * 252).std() / np.sqrt(time)
            #         # mu = logreturn.rolling(window=windowlen*252).mean()/time + (sigma**2)/2
            #         d1 = ((r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            #         # print(d1)
            #         if opt_type == "Call":
            #             delta = stat.norm.cdf(d1)
            #             vega = stat.norm.pdf(d1) * st * np.sqrt(T)
            #         else:
            #             delta = -stat.norm.cdf(-d1)
            #             vega = stat.norm.pdf(d1) * st * np.sqrt(T)
            #         v0 = euro_vanilla_put(st, st, T, r, sigma)
            #         option_loss = -((invest1 / st) * (st - st.shift(a)) + (invest2 / v0) * (
            #                     delta * (st - st.shift(a)) + vega * (sigma - sigma.shift(a))))
            #         VaR = []
            #         ES = []
            #         m = round(windowlen * 252 * (1 - 0.99))
            #         n = round(windowlen * 252 * (1 - 0.975))
            #         for i in range(windowlen * 252, len(option_loss) - 252 * windowlen):
            #             loss = sorted(option_loss[i:i + windowlen * 252], reverse=T)
            #             VaR.append(loss[m])
            #             ES.append(np.mean(loss[:n]))
            #         return VaR, ES
            #     var_option, es_option = his_optionloss(amount[0], amount[1], pricedata, T, r, window, horizon, opt_type="put")
            #
            #     return {'data': [go.Scatter(
            #         x=var_option.index,
            #         y=pd.Series(var_option.iloc[0]),
            #         # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            #         mode='lines+markers',
            #         name="VaR",
            #         marker={
            #             'size': 1,
            #             'opacity': 0.5,
            #             'line': {'width': 0.5, 'color': 'white'}
            #         }
            #     ),
            #         go.Scatter(
            #             x=es_option.index,
            #             y=pd.Series(es_option.iloc[1]),
            #             # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            #             mode='lines+markers',
            #             name="ES",
            #             marker={
            #                 'size': 1,
            #                 'opacity': 0.5,
            #                 'line': {'width': 0.5, 'color': 'white'}
            #             }),
            #
            #     ],
            #         'layout': go.Layout(
            #             xaxis={
            #                 'title': "Date",
            #             },
            #             yaxis={
            #                 'title': "Stock Price",
            #             },
            #             margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            #             hovermode='closest'
            #         )
            #     }


            if (method == "Monte Carlo"):
                def my_func(x):  # x is a slice
                    x = x.tolist()
                    # print(sorted(x,reverse=True)[:25])
                    stat = np.mean(sorted(x, reverse=True)[:25])
                    # print(stat)
                    return stat

                def euro_vanilla_put(S, K, T, r, sigma):

                    # S: spot price
                    # K: strike price
                    # T: time to maturity
                    # r: interest rate
                    # sigma: volatility of underlying asset

                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    put = (K * np.exp(-r * T) * stat.norm.cdf(-d2, 0.0, 1.0) - S * stat.norm.cdf(-d1, 0.0, 1.0))
                    return put

                def euro_vanilla_call(S, K, T, r, sigma):
                    # S: spot price
                    # K: strike price
                    # T: time to maturity
                    # r: interest rate
                    # sigma: volatility of underlying asset

                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    call = S * stat.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stat.norm.cdf(d2, 0.0, 1.0)
                    return call

                def mc_optionloss(s0, mu, sigma, invest_1, invest_2, T, r, windowlen, time, opt_type):
                    """time = 1/252, windowlen=5,T= 252"""
                    T = 5 / 252
                    # s0=price
                    # logreturn = np.log(price/price.shift(1))
                    # sigma = logreturn.rolling(window=windowlen*252).std()/np.sqrt(time)
                    # mu = logreturn.rolling(window=windowlen*252).mean()/time + (sigma**2)/2
                    # print(len(sigma))
                    w1 = stat.norm.ppf(np.random.rand(1000, )) * (T ** 0.5)
                    v0 = euro_vanilla_put(s0, s0, 1, r, sigma)
                    # w1 = np.array(w1*(T**0.5))
                    # print(w1)
                    # print(np.shape(w1))
                    st = {}
                    loss = {}
                    for i in range(len(w1)):
                        st[str(i)] = s0 * np.exp((mu - (sigma ** 2) / 2) * T + sigma * w1[i])
                        if opt_type == "Put":
                            vt = euro_vanilla_put(st[str(i)], s0, 1, r, sigma)
                            loss[str(i)] = (invest_1 / s0) * (s0 - st[str(i)]) + invest_2 / v0 * (v0 - vt)
                        elif opt_type == "Call":
                            vt = euro_vanilla_call(st[str(i)], s0, 1, r, sigma)
                            loss[str(i)] = invest_1 / s0 * (s0 - st[str(i)]) + invest_2 / vt * (vt - v0)

                    #st = pd.DataFrame(st)
                    loss = pd.DataFrame.from_dict(loss)
                    # print(st.head())
                    # print(loss.tail())
                    var = np.percentile(loss, 99, axis=1)
                    m = (1 - 0.975) * 1000
                    ES = loss.apply(my_func, axis=1)
                    return var, ES
                var,es = mc_optionloss(pricedata,mu,vol,amount[0],amount[1],T,r,window,horizon,"Put")



                #return u"{}".format(his_optionloss(amount[0], amount[1], pricedata, T, r, window, horizon, "put"))
                return {'data': [go.Scatter(
                    x=var.index,
                    y=pd.Series(var),
                    # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                    mode='lines+markers',
                    name="VaR",
                    marker={
                        'size': 1,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                    }
                ),
                     go.Scatter(
                                x=es.index,
                                y=pd.Series(es),
                                # text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                                mode='lines+markers',
                                name="ES",
                                marker={
                                    'size': 1,
                                    'opacity': 0.5,
                                    'line': {'width': 0.5, 'color': 'white'}
                                }),

                                 ],
                    'layout': go.Layout(
                        xaxis={
                            'title': "Date",
                        },
                        yaxis={
                            'title': "Stock Price",
                        },
                        margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                        hovermode='closest'
                    )
                }






@app.callback(
    Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return app_1_layout
    elif pathname == '/page-2':
        return app_2_layout
    elif pathname == '/page-3':
        return app_3_layout
    else:
        return index_page


if __name__ == '__main__':
   app.run_server(debug=True)
   #server = Flask(__name__)
   app.run_server(debug=False, host='0.0.0.0')