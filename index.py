import dash
#import dash_table
from dash import dcc, html, dash_table
#import dash_core_components as dcc
import dash_bootstrap_components as dbc
#import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from fanalysis.pca import PCA
import dash_extensions as de
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

#----------------------------------------- Données---------------------------------------------------

df = pd.read_csv("assets/decathlon.txt", sep='\t')

# Eliminer les colonnes dont nous n'aurons pas besoin
data = df.drop(['Points', 'Rank', 'Competition'], axis=1)

df.index.rename('Individus', inplace=True)

df = df.reset_index(level=0)

#----------------------------------------------------------- Calculs------------------------------------------------------

nb_ligne = data.shape[0]

nb_col = data.shape[1]

individus = data.columns

missdonnees = [data[i].isna().sum() for i in data.columns]

typedonnees = [data[i].dtype for i in data.columns]

info_variables = pd.DataFrame([individus, typedonnees, pd.to_numeric(missdonnees)]).T

info_variables.rename(columns={0: 'Variables', 1: 'Type', 2: 'Missing'}, inplace=True)

tab_val = round(data.describe(), 2)

tab_val["Stats"] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

tab_val = pd.DataFrame(tab_val)

tab_val.set_index("Stats", inplace=True)

tab_val = tab_val.reset_index(level=0)

df_corr = df[['100m', 'Long.jump', 'Shot.put', 'High.jump', '400m','110m.hurdle', 'Discus', 'Pole.vault', 'Javeline', '1500m']]

corr_tab_val = df_corr.corr(method='pearson')

#---------------------------------------------------------GRAPHIQUES---------------------------------------------------

x = list(corr_tab_val.columns)
y = list(corr_tab_val.index)
z = np.array(corr_tab_val)

fig_corr = ff.create_annotated_heatmap(
    z,
    x=x,
    y=y,
    annotation_text=np.around(z, decimals=2),
    hoverinfo='z',
    colorscale='sunset', showscale=True
    )

lis_col = ['100m', 'Long.jump', 'Shot.put', 'High.jump', '400m', '110m.hurdle', 'Discus', 'Pole.vault', 'Javeline', '1500m']

#----------------------------------------- A C P --------------------------------------------------------------------
X = data.to_numpy()
my_pca = PCA(std_unit=True, row_labels=data.index.values, col_labels=data.columns.values)
my_pca.fit(X)
abs_val_prop = my_pca.eig_[0]
pct_val_prop = my_pca.eig_[1]
pct_cumul_val_prop = my_pca.eig_[2]
val_prop = pd.DataFrame(list(zip(abs_val_prop, pct_val_prop, pct_cumul_val_prop)),
               columns =['Valeurs propres', 'Pourcentages', 'Pourcentages cumulés'])

val_prop = round(val_prop, 3)

df_rows = my_pca.row_topandas()
coord_ind = round(df_rows.iloc[:, 0:10], 3)
cont_ind = round(df_rows.iloc[:, 10:20], 3)
cos2_ind = round(df_rows.iloc[:, 20:30], 3)
coord_ind = coord_ind.reset_index()
cont_ind = cont_ind.reset_index()
cos2_ind = cos2_ind.reset_index()

dim_ind_name = ["Individus", "Dim1", "Dim2", "Dim3", "Dim4", "Dim5", "Dim6", "Dim7", "Dim8", "Dim9", "Dim10"]
coord_ind.columns = dim_ind_name
cont_ind.columns = dim_ind_name
cos2_ind.columns = dim_ind_name

df_cols = my_pca.col_topandas()
coord_var = round(df_cols.iloc[:, 0:10], 3)
cont_var = round(df_cols.iloc[:, 10:20], 3)
cos2_var = round(df_cols.iloc[:, 20:30], 3)
coord_var = coord_var.reset_index()
cont_var = cont_var.reset_index()
cos2_var = cos2_var.reset_index()


dim_var_name = ["Variables", "Dim1", "Dim2", "Dim3", "Dim4", "Dim5", "Dim6", "Dim7", "Dim8", "Dim9", "Dim10"]

coord_var.columns = dim_var_name
cont_var.columns = dim_var_name
cos2_var.columns = dim_var_name

liste_axe = ["Axe1", "Axe2", "Axe3", "Axe4", "Axe5", "Axe6", "Axe7", "Axe8", "Axe9", "Axe10"]

#------------------------------------------------------Images--------------------------------------------------------------

#img_speed ='assets/20534-speedometer.json'

img_info = 'assets/68575-icon-of-information.json'

img_dash = 'assets/75082-dashboard.json'

img_data = 'assets/83426-database.json'

#----------------------------------------------------------STYLE------------------------------------------------------------

tabs_styles = {
    'height': '44px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

#----------------------------------------- Application---------------------------------------------------
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

#---------------------------------------------------- Page d'acceuil----------------------------------------------------------
accueil = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H1("ACP-DASHBOARD", style={'border-radius': '10px', 'background': 'aliceblue', 'color': 'navy'}),
                    html.Br(), html.Br(),
                    html.Button(
                        html.H5("START"),
                        id="ouvrir",
                        n_clicks=0,
                        style={
                            'border-radius': '5px',
                            'border': 'solid mediumblue',
                            'box-shadow': '2px 3px navy',
                            'color': '#00CED1',
                            'background': 'aliceblue',
                            'width': '300px',
                            'height': '30px',
                            'margin-top': '30px'
                        }
                    )
                ], style={'text-align': 'center'})
            ], style={'margin-top': '100px'}),
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("Evaluation cours de : ", style={'margin-bottom': '5px'}),
                    html.P("Data mining et visualisation", style={'margin-bottom': '5px'}),

                ], style={'text-align': 'center', 'font-size': '0.7rem', 'color': 'darkslategray'}),
                dbc.Col([
                    html.Hr(),
                    html.P("Auteurs : ", style={'margin-bottom': '5px'}),
                    html.P("Assana Richard AYIZOU", style={'margin-bottom': '5px'}),
                    html.P("Jean Jacques Roger FAYE,", style={'margin-bottom': '5px'}),
                    html.P("ISE2 : 2021-2022")
                ], style={'text-align': 'center', 'font-size': '0.7rem', 'color': 'darkslategray'}),
                dbc.Col([
                    html.Hr(),
                    html.P("Enseignant : ", style={'margin-bottom': '5px'}),
                    html.P("M. Aliou TINE,", style={'margin-bottom': '5px'}),
                    html.P("Ingénieur Statisticien Economiste")
                ], style={'text-align': 'center', 'font-size': '0.7rem', 'color': 'darkslategray'})
            ], style={'margin-top': '100px'})
        ])
    ], style={'padding': '10px 100px'}),
])

#----------------------------------------------AFICHAGE DES DONNEES--------------------------------------------------------
page_data = html.Div([
    dbc.Row([
        dbc.Col([

            html.Hr(),
            html.H5("Dimensions de la base", style={'color': '#4169E1'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Nombre de variables"), style={'background': '#4169E1', 'color': 'white'}),
                        dbc.CardBody(html.H6(nb_col))
                    ])
                ], style={'text-align': 'center'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Nombre de lignes"), style={'background': '#4169E1', 'color': 'white'}),
                        dbc.CardBody(html.H6(nb_ligne))
                    ])
                ], style={'text-align': 'center'})

            ])
        ], width=4, style={'margin': '2px 18px'}),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([html.H5("Base de données")], style={'background': '#4169E1', 'color': 'white'}),
                        dbc.CardBody([
                            html.Div([
                                dash_table.DataTable(data=df.to_dict("records"),
                                                     columns=[{'name': i, 'id': i} for i in df.columns],
                                                     fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                                     style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',},
                                                     style_data={'color': 'black', 'backgroundColor': 'white', 'font-family': 'calibri'},
                                                     style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF'}],
                                                     style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold', 'font-family': 'calibri'},
                                                     style_table={'height': '400px', 'minWidth': '100%', 'overflowY': 'auto','overflowX': 'auto'})
                            ])
                        ])
                    ])
                ], style={'text-align': 'center'})
            ], style={'margin-top': '10px'})
        ], width=7)
    ], style={'font-family': 'calibri'})
])

#----------------------------------------------DESCRIPTION DES DONNEES----------------------------------------------------
page_desc = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Summary", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=tab_val.to_dict('records'),
                                    columns=[{'name': i,'id': i} for i in tab_val.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '300px', 'minWidth': '100%', 'overflowY': 'auto','overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Corrélations", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig_corr
                    )
                ])
            ])
        ], style=tab_style)
    ], style=tabs_styles)
])

#------------------------------------------VISUALISATION DES DONNEES------------------------------------------------------
page_viz = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Distributions", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Choisir une variable"),
                            dcc.Dropdown(
                                options=[{'label': i, 'value': i} for i in data.columns],
                                id='dd-viz-variable',
                                value=data.columns[0]
                            )
                        ])
                    ]), html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(de.Lottie(options=options, url=img_info, width="20%", height="20%")),
                                dbc.CardBody([
                                    html.H6(id="viz-count",),
                                    html.H6(id="viz-mean",),
                                    html.H6(id="viz-std",),
                                    html.H6(id="viz-min",),
                                    html.H6(id="viz-25pct",),
                                    html.H6(id="viz-50pct",),
                                    html.H6(id="viz-75pct",),
                                    html.H6(id="viz-max",), html.Br(), html.Br()
                                ], style={'text-align': 'left'})
                            ])
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id="viz-box")
                                ], style={'margin-bottom': '30px'})
                            ])
                        ]),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id="viz-hist")
                                ], style={'margin-bottom': '30px'})
                            ])
                        ])
                    ])
                ])
            ], style={'margin': '20px'})
        ], style=tab_style),
        dcc.Tab(label="Interactions", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir variables"),
                    dcc.Dropdown(
                        id="dd-interactions",
                        options=[{'label': i, 'value': i} for i in lis_col],
                        multi=True,
                        value=['100m', 'Long.jump', 'Shot.put']
                    ),
                    dcc.Graph(id="grah-interactions")
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Interactions/Compétition", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir variables"),
                    dcc.Dropdown(
                        id="dd-interactions2",
                        options=[{'label': i, 'value': i} for i in lis_col],
                        multi=True,
                        value=['100m', 'Long.jump', 'Shot.put']
                    ),
                    dcc.Graph(id="grah-interactions2")
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Graph 3D", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Axe1"),
                    dcc.Dropdown(
                        id="dd-3D-interact1",
                        options=[{'label': i, 'value': i} for i in lis_col],
                        value=lis_col[0]
                    ),
                    html.H6("Axe2"),
                    dcc.Dropdown(
                        id="dd-3D-interact2",
                        options=[{'label': i, 'value': i} for i in lis_col],
                        value=lis_col[1]
                    ),
                    html.H6("Axe3"),
                    dcc.Dropdown(
                        id="dd-3D-interact3",
                        options=[{'label': i, 'value': i} for i in lis_col],
                        value=lis_col[2]
                    )
                ], width=3),
                dbc.Col([dcc.Graph(id="3D-grah-interact")])
            ])
        ], style=tab_style)
    ], style=tabs_styles)
])

#------------------------------------------------ ACP : VALEURS PROPRES ----------------------------------------------------
page_val_prop = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Tableau des valeurs propres", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=val_prop.to_dict('records'),
                                    columns = [{'name': i,'id': i} for i in val_prop.columns],
                                    fixed_rows={'headers': True},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '330px', 'overflowY': 'auto','overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Graphiques des valeurs propres", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Valeurs propres"),
                    dcc.Graph(
                        figure=px.bar(val_prop, y=abs_val_prop)
                    )
                ]),
                dbc.Col([
                    html.H6("Pourcentages"),
                    dcc.Graph(
                        figure=px.bar(val_prop, y=pct_val_prop, color=pct_val_prop)
                    )
                ]),
                dbc.Col([
                    html.H6("Pourcentages cumulés"),
                    dcc.Graph(
                        figure=px.bar(val_prop, y=pct_cumul_val_prop, color=pct_cumul_val_prop)
                    )
                ])
            ])
        ], style=tab_style)
    ], style=tabs_styles)
])

#----------------------------------------- ACP : RESULTATS ANALYSE DES INDIVIDUS----------------------------------------------
page_individus = html.Div([
    dbc.Row([
            dbc.Col([
                html.H5("ANALYSE DES INDIVIDUS DE L'ACP")
            ], style={'text-align': 'center', 'background': 'blue', 'color': 'white', 'margin': '2px 18px','box-shadow': '2px 3px navy'})
    ], style={'margin-bottom': '15px'}),
    dcc.Tabs([
        dcc.Tab(label="Coordonnées", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=coord_ind.to_dict('records'),
                                    columns = [{'name': i, 'id': i} for i in coord_ind.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto','overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Contributions", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=cont_ind.to_dict('records'),
                                    columns = [{'name': i, 'id': i} for i in cont_ind.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF'}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto', 'overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Cos2", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=cos2_ind.to_dict('records'),
                                    columns = [{'name': i,'id': i} for i in cos2_ind.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto', 'overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style)
    ], style=tabs_styles)
])

#----------------------------------------- ACP : RESULTATS ANALYSE DES VARIABLES----------------------------------------------
page_variables = html.Div([
    dbc.Row([
            dbc.Col([
                html.H5("ANALYSE DES VARIABLES DE L'ACP")
            ], style={'text-align': 'center', 'background': 'blue', 'color': 'white', 'margin': '2px 18px','box-shadow': '2px 3px navy'})
    ], style={'margin-bottom': '15px'}),
    dcc.Tabs([
        dcc.Tab(label="Coordonnées", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=coord_var.to_dict('records'),
                                    columns = [{'name': i,'id': i} for i in coord_var.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto', 'overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Contributions", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=cont_var.to_dict('records'),
                                    columns = [{'name': i,'id': i} for i in cont_var.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF',}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto', 'overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style),
        dcc.Tab(label="Cos2", children=[
            dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dash_table.DataTable(
                                    data=cos2_var.to_dict('records'),
                                    columns = [{'name': i,'id': i} for i in cos2_var.columns],
                                    fixed_rows={'headers': True}, fixed_columns={'headers': True, 'data': 1},
                                    style_cell={'textAlign': 'center', 'minWidth': '140px', 'width': '140px', 'maxWidth': '140px'},
                                    style_data={'color': 'black', 'backgroundColor': 'white'},
                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8FF'}],
                                    style_header={'backgroundColor': '#4169E1', 'color': 'white', 'fontWeight': 'bold'},
                                    style_table={'height': '360px', 'minWidth': '100%', 'overflowY': 'auto', 'overflowX': 'auto'}
                                )
                            ])
                        ])
                    ])
                ], style={'margin-top': '15px'})
        ], style=tab_style)
    ], style=tabs_styles)

])

#----------------------------------------- ACP : GRAPHIQUES----------------------------------------------
page_graphiques = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Cercle des corrélations", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Choisir premier axe"),
                            dcc.Dropdown(
                                options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                                id='acp-axe-1',
                                value=1
                            )
                                ])
                    ], style={'margin-bottom': '15px'}),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Choisir deuxième axe"),
                            dcc.Dropdown(
                                options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                                id='acp-axe-2',
                                value=2
                            )
                        ])
                    ]), html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(de.Lottie(options=options, url=img_info, width="20%", height="20%")),
                                dbc.CardBody([
                                    html.H6(id="pct-axe1-cercle",),
                                    html.H6(id="pct-axe2-cercle",),
                                    html.H6(id="pct-total-cercle",)
                                ], style={'text-align': 'left', 'font-family': 'calibri', 'color': '#4169E1'})
                            ])
                        ])
                    ])
                ], width=3, style={'padding-top': '50px'}),
                dbc.Col([
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([dcc.Graph(id="coor-circle")])
                        ], style={'width': '100%'})
                    ])
                ], style={'text-align': 'center'})
            ])
        ], style=tab_style),
        dcc.Tab(label="Nuages des individus", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Choisir premier axe"),
                            dcc.Dropdown(
                                options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                                id='acp-ind-axe-1',
                                value=1
                            )
                                ])
                    ], style={'margin-bottom': '15px'}),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Choisir deuxième axe"),
                            dcc.Dropdown(
                                options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                                id='acp-ind-axe-2',
                                value=2
                            )
                                ])
                    ]), html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(de.Lottie(options=options, url=img_info, width="20%", height="20%")),
                                dbc.CardBody([
                                    html.H6(id="pct-axe1-nuage",),
                                    html.H6(id="pct-axe2-nuage",),
                                    html.H6(id="pct-total-nuage",)
                                ], style={'text-align': 'left', 'font-family': 'calibri', 'color': '#4169E1'})
                            ])
                        ])
                    ])
                ], width=3, style={'padding-top': '50px'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="nuage-individus")])
                    ])
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Contributions individus", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir un axe"),
                    dcc.Dropdown(
                        options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                        id='acp-contind-axe',
                        value=1
                    )
                ], width=4, style={'padding-top': '50px'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="contributions-ind")])
                    ])
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Cos2 individus", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir axe"),
                    dcc.Dropdown(
                        options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                        id='acp-cosind-axe',
                        value=1
                    )
                ], width=4, style={'padding-top': '50px'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="cos2-ind")])
                    ])
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Contributions variables", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir un axe"),
                    dcc.Dropdown(
                        options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                        id='acp-contvar-axe',
                        value=1
                    )
                ], width=4, style={'padding-top': '50px'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([dcc.Graph(id="contributions-var")])
                    ])
                ])
            ])
        ], style=tab_style),
        dcc.Tab(label="Cos2 variables", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Choisir un axe"),
                    dcc.Dropdown(
                        options=[{'label': i, 'value': j} for i, j in list(zip(liste_axe, range(1, 10)))],
                        id='acp-cos2var-axe',
                        value=1
                    )
                ], width=4, style={'padding-top': '50px'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="cos2-var")])
                    ])
                ])
            ])
        ], style=tab_style)
    ], style={'height': '55px'})
])

#----------------------------------------- PAGE PRINCIPALE----------------------------------------------
page_principale = html.Div([
    html.Div([
        dbc.Row([dbc.Col([
            dbc.Nav([
                dbc.NavLink("Données", href="/page-data"),
                dbc.NavLink("Description", href="/page-desc"),
                dbc.NavLink("Visualisation", href="/page-viz"),
                dbc.NavLink("ACP-Valeurs propres", href="/page-val-prop"),
                dbc.NavLink("ACP-Individus", href="/page-acp-ind"),
                dbc.NavLink("ACP-Variables", href="/page-acp-var"),
                dbc.NavLink("ACP-Graphiques", href="/page-acp-graph")
            ], pills=True)
        ], style={'text-align': 'left', 'font-family': 'calibri','color': 'white'})]),
        dbc.Row([
            dbc.Col([
            ], style={'background': 'blue', 'margin': '2px 5px','box-shadow': '2px 2px navy'})
        ], style={'height': '6px'})
    ], style={
        'position': 'fixed',
        'top': 2,
        'left': 2,
        'height': '60px',
        'right': 2,
        'border-radius': '3px'
    }),
    html.Div(
        id="page-principale",
        style={
        'position': 'fixed',
        'top': 65,
        'left': 2,
        'bottom': 2,
        'right': 2,
        'border-radius': '3px'
    })
])



container = html.Div(
    id="container",
    children=accueil,
    style={
        'position': 'fixed',
        'left': 2,
        'right': 2,
        'bottom': 2,
        'top': 2
    }
)

app.layout = html.Div(children=[
    dcc.Location(id="url"),
    container
])


#------------------------------------------------ OUVRIR LE DASHBOARD--------------------------------------------------------
@app.callback(
    Output('container', 'children'),
    Output('url', 'pathname'),
    Input('ouvrir', 'n_clicks')
)
def ouvrir_app(clic):
    if clic == 0:
        raise dash.exceptions.PreventUpdate
    else:
        lien = '/page-data'
        return page_principale, lien


#---------------------------------------------------- CHARGEMENT DES PAGES ---------------------------------------------------------
@app.callback(
    Output('page-principale', 'children'),
    Input('url', 'pathname')
)
def load_page(lien):
    if lien=='/':
        raise dash.exceptions.PreventUpdate
    elif lien=='/page-data':
        return page_data
    elif lien=='/page-desc':
        return page_desc
    elif lien=='/page-viz':
        return page_viz
    elif lien=='/page-val-prop':
        return page_val_prop
    elif lien=='/page-acp-ind':
        return page_individus
    elif lien=='/page-acp-var':
        return page_variables
    elif lien=='/page-acp-graph':
        return page_graphiques
    return "Error 404 : Page not found !"

#--------------------------------------------------------Page Visualisation------------------------------------------------
@app.callback(
    Output(component_id='viz-count', component_property='children'),
    Output(component_id="viz-mean", component_property="children"),
    Output(component_id="viz-std", component_property="children"),  #
    Output(component_id="viz-min", component_property="children"),
    Output(component_id="viz-25pct", component_property="children"),
    Output(component_id="viz-50pct", component_property="children"),
    Output(component_id="viz-75pct", component_property="children"),
    Output(component_id="viz-max", component_property="children"),
    Output(component_id="viz-box", component_property="figure"),
    Output(component_id="viz-hist", component_property="figure"),

    Input(component_id='dd-viz-variable', component_property='value')
)
def load_ch_ref(variable):
    nb_val = "Observations : {}".format(data[variable].shape[0])
    moyenne = "Moyenne : {}".format(round(data[variable].mean(), 3))
    ecart_type = "Ecart type : {}".format(round(data['100m'].std(), 3))
    minimum = "Minimum : {}".format(data[variable].min())
    v_25pct = "25% : {}".format(data[variable].quantile(0.25))
    mediane = "Médiane : {}".format(data[variable].quantile(0.5))
    v_75pct = "75% : {}".format(data[variable].quantile(0.75))
    maximum = "Maximum : {}".format(data[variable].max())
    box = px.box(data, y=variable)
    hist = px.histogram(data, x=variable)
    return nb_val, moyenne, ecart_type, minimum, v_25pct, mediane, v_75pct, maximum, box, hist


@app.callback(
    Output("grah-interactions", "figure"),
    Input("dd-interactions", "value")
)
def update_graph_interact(variables):
    return px.scatter_matrix(df, dimensions=variables)


@app.callback(
    Output("grah-interactions2", "figure"),
    Input("dd-interactions2", "value")
)
def update_graph_interact(variables):
    return px.scatter_matrix(df, dimensions=variables, color="Competition")

@app.callback(
    Output("3D-grah-interact", "figure"),
    Input("dd-3D-interact1", "value"),
    Input("dd-3D-interact2", "value"),
    Input("dd-3D-interact3", "value")
)
def update_3Dgraph_interact(axe1,axe2, axe3):
    return px.scatter_3d(df, x=axe1, y=axe2, z=axe3,
              color='Competition')


#------------------------------------------------CERCLE DE CORRELATION-----------------------------------------------------

@app.callback(
    Output('coor-circle', 'figure'),
    Input('acp-axe-1', 'value'),
    Input('acp-axe-2', 'value')
)
def update_circle(axe1, axe2):
    fig_cercle = go.Figure()
    fig_cercle.update_xaxes(range=[-1.2, 1.2], zeroline=False)
    fig_cercle.update_yaxes(range=[-1.2, 1.2])
    fig_cercle.add_shape(
        dict(type="line", x0=-1.2, x1=1.2, y0=0, y1=0, line_color="gray")
    )
    fig_cercle.update_shapes(line_dash="dash")
    fig_cercle.add_shape(
        dict(type="line", x0=0, x1=0, y0=-1.2, y1=1.2, line_color="gray")
    )
    fig_cercle.update_shapes(line_dash="dash")
    fig_cercle.add_shape(
        dict(type="rect", x0=-1.2, x1=1.2, y0=-1.2, y1=1.2, line_color="navy")
    )
    fig_cercle.add_shape(type="circle",
                         xref="x", yref="y",
                         x0=-1, y0=-1, x1=1, y1=1,
                         line_color="gray",
                         )
    fig_cercle.update_layout(width=500, height=500)

    x_coord = coord_var.iloc[:, axe1]
    y_coord = coord_var.iloc[:, axe2]
    text_label = ['100m', 'Long.jump', 'Shot.put', 'High.jump', '400m', '110m.hurdle',
                  'Discus', 'Pole.vault', 'Javeline', '1500m']

    fig_cercle.add_trace(go.Scatter(
        x=x_coord,
        y=y_coord,
        text=text_label,
        mode="text",
    ))

    for i, j in list(zip(x_coord, y_coord)):
        fig_cercle.add_shape(
            dict(type="line", x0=0, x1=i, y0=0, y1=j, line_color="black")
        )

    return fig_cercle

@app.callback(
    Output("pct-axe1-cercle", "children"),
    Output("pct-axe2-cercle", "children"),
    Output("pct-total-cercle", "children"),
    Input('acp-axe-1', 'value'),
    Input('acp-axe-2', 'value')
)
def update_pct_dim_cercle(axe1, axe2):
    pct_axe1=pct_val_prop[axe1]
    pct_axe2 = pct_val_prop[axe2]
    pct_total = pct_val_prop[axe1]+pct_val_prop[axe2]
    return [
        html.H6("Dimension {} : ".format(axe1)+"{}".format(round(pct_axe1, 2))+"%"),
        html.H6("Dimension {} : ".format(axe2)+"{}".format(round(pct_axe2, 2))+"%"),
        html.H6("Total : {} ".format(round(pct_total, 2))+"%")
    ]


#---------------------------------------------------NUAGE INDIVIDUS-----------------------------------------------------
@app.callback(
    Output("nuage-individus", "figure"),
    Input('acp-ind-axe-1', 'value'),
    Input('acp-ind-axe-2', 'value')
)
def update_nuage_individus(axe1, axe2):
    fig_nuage_ind = go.Figure()
    fig_nuage_ind.update_xaxes(range=[-5, 5], zeroline=False)
    fig_nuage_ind.update_yaxes(range=[-5, 5])
    fig_nuage_ind.add_shape(
        dict(type="line", x0=-5, x1=5, y0=0, y1=0, line_color="gray")
    )
    fig_nuage_ind.update_shapes(line_dash="dash")
    fig_nuage_ind.add_shape(
        dict(type="line", x0=0, x1=0, y0=-5, y1=5, line_color="gray")
    )
    fig_nuage_ind.update_shapes(line_dash="dash")
    fig_nuage_ind.add_shape(
        dict(type="rect", x0=-5, x1=5, y0=-5, y1=5, line_color="navy")
    )

    x_coord = coord_ind.iloc[:, axe1]
    y_coord = coord_ind.iloc[:, axe2]
    text_label = coord_ind["Individus"]

    fig_nuage_ind.add_trace(go.Scatter(
        x=x_coord,
        y=y_coord,
        text=text_label,
        mode="text",
        textfont=dict(color="red")
    ))

    return fig_nuage_ind


@app.callback(
    Output("pct-axe1-nuage", "children"),
    Output("pct-axe2-nuage", "children"),
    Output("pct-total-nuage", "children"),
    Input('acp-ind-axe-1', 'value'),
    Input('acp-ind-axe-2', 'value')
)
def update_pct_dim_nuage(axe1, axe2):
    pct_axe1 = pct_val_prop[axe1]
    pct_axe2 = pct_val_prop[axe2]
    pct_total = pct_val_prop[axe1]+pct_val_prop[axe2]
    return [
        html.H6("Dimension {} : ".format(axe1)+"{}".format(round(pct_axe1, 2))+"%"),
        html.H6("Dimension {} : ".format(axe2)+"{}".format(round(pct_axe2, 2))+"%"),
        html.H6("Total : {} ".format(round(pct_total, 2))+"%")
    ]

#-------------------------------------------------CONTRIBUTION DES INDIVIDUS----------------------------------------------
@app.callback(
    Output("contributions-ind", "figure"),
    Input('acp-contind-axe', 'value')
)
def update_cont_ind(axe):
    nom_col=cont_ind.columns[axe]
    contribution = cont_ind.sort_values(by=nom_col)
    return px.bar(contribution, x=contribution.iloc[:, axe], y='Individus')

#--------------------------------------------------- COS2 INDIVIDUS-----------------------------------------------------
@app.callback(
    Output("cos2-ind", "figure"),
    Input('acp-cosind-axe', 'value')
)
def update_cos2_ind(axe):
    nom_col = cos2_ind.columns[axe]
    cos2 = cos2_ind.sort_values(by=nom_col)
    return px.bar(cos2, x=cos2.iloc[:, axe], y='Individus')


#------------------------------------------------------ CONTRIBUTIONS VARIABLES--------------------------------------------
@app.callback(
    Output("contributions-var", "figure"),
    Input('acp-contvar-axe', 'value')
)
def update_cont_var(axe):
    nom_col = cont_var.columns[axe]
    contribution = cont_var.sort_values(by=nom_col)
    return px.bar(contribution, x=contribution.iloc[:, axe], y='Variables')

#----------------------------------------------COS2 VARIABLES---------------------------------------------------------
@app.callback(
    Output("cos2-var", "figure"),
    Input('acp-cos2var-axe', 'value')
)
def update_cos2_var(axe):
    nom_col = cos2_var.columns[axe]
    cos2 = cos2_var.sort_values(by=nom_col)
    return px.bar(cos2, x=cos2.iloc[:, axe], y='Variables')

#-----------------------------------------------run app---------------------------------------------------
app.title="AYIZOU & FAYE"
app.run_server(debug=True)