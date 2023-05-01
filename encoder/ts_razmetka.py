import os
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import ctx
import dash_html_components as html
import dash_core_components as dcc
from datetime import datetime
import pickle
import pandas as pd
import plotly.express as px
import pickle
import numpy as np


# Define the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])


class TsType():
    TEMP = 0
    OBJ_ALL = 1


class DataWork():
    def __init__(self, data_path, ts_type=TsType.TEMP):
        self.ts_type = ts_type
        with open(data_path, 'rb') as f:
            self.dd = pickle.load(f)
        self.id_list = list(self.dd.keys())
        self.current_id = 0
        self.classes = self.init_classes(data_path)

    def init_classes(self, datapath):
        class_path = os.path.dirname(datapath)
        self.class_file = os.path.join(class_path, 'ts_classes.pkl')
        if os.path.isfile(self.class_file):
            with open(self.class_file, 'rb') as f:
                classes = pickle.load(f)
        else:
            classes = {}
            for key in self.id_list:
                classes[key] = 0
        return classes

    def generate_fig(self):
        if self.ts_type == TsType.TEMP:
            df = self.dd[self.id_list[self.current_id]]
            fig = px.line(df, x="DUR", y="TEMP_delta")
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
            return fig

    def next_id(self):
        if self.current_id != len(self.id_list)-1:
            self.current_id += 1
        else:
            self.current_id = 0
        return self.current_id

    def prev_id(self):
        if self.current_id != 0:
            self.current_id -= 1
        else:
            self.current_id = len(self.id_list)-1
        return self.current_id

    def generate_options(self):
        return [{'label': f, 'value': f} for f in self.id_list]

    def save_classes(self):
        with open(self.class_file, 'wb') as f:
            pickle.dump(self.classes, f)


dw = DataWork(
    '/home/roman/projects/pytorch-video-pipeline/dataset_ts/dataset3.pkl', TsType.TEMP)


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


# Define the layout
app.layout = dbc.Container([
    html.H1('TS Explorer'),
    dbc.Row([
        dbc.Col([

            html.Hr(),
            html.Label('Select class'),
            html.Br(),
            dbc.RadioItems(
                id='class-selection',
                options=[
                    {'label': 'Norm', 'value': 0},
                    {'label': 'with processes', 'value': 1},
                    {'label': 'Anomaly', 'value': 2},
                ],
                value=dw.classes[dw.id_list[dw.current_id]]
            ),
            html.Hr(),
            html.Br(),
            dcc.Dropdown(
                id='id-selection',
                options=dw.generate_options(),
                value=dw.id_list[dw.current_id]
            ),
            html.Hr(),
            html.Br(),
            dbc.Button('Previous', id='prev-button', color='primary',
                       className='mr-1', style={'margin-right': '10px', 'margin-left': '10px'}),
            dbc.Button('Next', id='next-button',
                       color='primary', className='mr-1'),
            html.Hr(),
            html.Br(),
            dbc.Button('Save', id='save-button',
                       color='primary', className='mr-1', style={'margin-left': '10px'}),
            html.Div(id='save_text'),
            html.Div(id='save_class')
        ], md=3),
        dbc.Col([
            dcc.Graph(
                id='ts-graph-1',
                figure=dw.generate_fig()
            )
        ], md=9)
    ])
], fluid=True)


@app.callback(
    Output('class-selection', 'value'),
    Input('id-selection', 'value'),
)
def update_class(selected_id):
    return dw.classes[selected_id]


@app.callback(
    Output('save_class', 'children'),
    Input('class-selection', 'value'),
)
def save_class(selected_class):
    print(dw.current_id)
    dw.classes[dw.id_list[dw.current_id]] = selected_class
    return 'class_changed'


@app.callback(
    Output('id-selection', 'value'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks')

)
def update_class_selection(prev_clicks, next_clicks):
    global current_id
    global id_list
    triggered_id = ctx.triggered_id
    if triggered_id == 'prev-button':
        if prev_clicks is not None:
            current_id = dw.prev_id()
            print(prev_clicks)

    elif triggered_id == 'next-button':
        if next_clicks is not None:
            current_id = dw.next_id()

    return dw.id_list[dw.current_id]


@app.callback(
    Output('save_text', 'children'),
    Input('save-button', 'n_clicks'),


)
def save_selection(save_clicks):
    if save_clicks is not None:
        dw.save_classes()
        return 'Saved'


@app.callback(
    Output('ts-graph-1', 'figure'),
    Input('id-selection', 'value'),


)
def update_class_selection(selected_id):
    print(selected_id)
    dw.current_id = np.where(np.array(dw.id_list) == selected_id)[0][0]
    fig = dw.generate_fig()

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
