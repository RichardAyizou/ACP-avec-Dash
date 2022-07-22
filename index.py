import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from fanalysis.pca import PCA
import dash_extensions as de
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

#----------------------------------------- Données---------------------------------------
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

info_variables.rename(columns={0:'Variables', 1:'Type', 2:'Missing'}, inplace=True)

tab_val = round(data.describe(),2)

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
    y=y ,
    annotation_text = np.around(z, decimals=2),
    hoverinfo='z',
    colorscale='sunset', showscale = True
    )

lis_col = ['100m', 'Long.jump', 'Shot.put', 'High.jump', '400m','110m.hurdle', 'Discus', 'Pole.vault', 'Javeline', '1500m']

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

img_speed ='assets/20534-speedometer.json'

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
