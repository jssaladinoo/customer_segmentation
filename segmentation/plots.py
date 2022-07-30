#!/usr/bin/env python
# coding: utf-8

# # Preprocessing module 

import os
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

palette = ['#731331', '#BD651A', '#DBAE58','#208288', '#0F2254'] 

pio.templates["tspalette"] = go.layout.Template(
    layout=go.Layout(
        colorscale = {'sequential': 'YlOrRd',  
                      'diverging': 'YlOrRd', 
                      'sequentialminus' : 'YlOrRd',
                     }, 
        colorway = ['#731331', '#BD651A', '#DBAE58','#208288', '#0F2254'][::-1]
    )
)

pio.templates.default = 'tspalette'


import logging, joblib
logger = logging.getLogger(__name__)


def groupings(data, groupings, perplexity = 'auto', 
              show_image = True, 
              model_name = None, 
              dataname = None, 
              path = None):  
    
    
    if perplexity == "auto": 
        perplexity = data.size**.5

    transform = TSNE(n_components = 2, 
        perplexity = perplexity, 
        random_state = 5
        ).fit_transform(data)
    
    transform = pd.DataFrame(transform, 
                    columns = ['feature_{}'.format(i) for i in range(2)])
    transform['date'] = data.index
    transform ['groupings'] = groupings
    
    non_outlier_transform = transform[transform.groupings != '-1']
    fig = px.scatter(non_outlier_transform, 
                    x = non_outlier_transform['feature_0'], 
                    y = non_outlier_transform['feature_1'],
                    color = 'groupings',  
                    color_discrete_sequence = ['#BD651A', '#DBAE58','#208288', '#0F2254', 
                                              'dodgerblue', 'turquoise',  'chartreuse'],
                    opacity = 0.6)
    fig.update_traces(marker_symbol = 'circle-open-dot',)
    
    fig.update_xaxes(showgrid = False, zeroline = False)
    fig.update_yaxes(showgrid = False, zeroline = False)
    fig.update_layout(legend =dict( orientation="h", 
                                yanchor="bottom", y=-0.3,
                                xanchor="center", x=0.5
                                ), 
                      title = '{} : Clustered using {}'.format(dataname, model_name )
                     )

    if show_image: 
        fig.show()
        
    if path is not None: 

        fig.write_image(os.path.join(path, 'cluster_visualization_{}.png'.format(model_name)))
        fig.write_html(os.path.join(path, 'cluster_visualization_{}.html'.format(model_name)))

        
        
def plot_cluster_means(cluster_means, categories, 
            show_image = True, 
            model_name = None, 
            dataname = None, 
            path = None) :

    fig = go.Figure()

    for cluster in cluster_means.index: 

        fig.add_trace(go.Scatterpolar(
              r= cluster_means.loc[cluster, :],
              theta=categories,
              fill='toself',
              name='{}'.format(cluster)
        ))


    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
        )),
      showlegend=True
    )

    if show_image: 
        fig.show()
        
    if path is not None: 

        fig.write_image(os.path.join(path, 'cluster_means_{}.png'.format(model_name)))
        fig.write_html(os.path.join(path, 'cluster_means_{}.html'.format(model_name)))
        
def anomaly_score(score_ts, threshold, show_image = True, model_name = None, path = None): 

    fig = go.Figure()

    fig.add_hline(y=threshold, 
                  line_width=1,  
                  line_color="DarkSlateGrey",
                  line_dash = 'dash', 
                  annotation_text = 'threshold = {}'.format(np.round(threshold, 2)), 
                  annotation_position = 'top left', 
                  annotation_bgcolor = 'gray'
        )
   
    normal_Xscores = score_ts[score_ts['score'] < threshold]

    fig.add_trace(
        go.Scatter(
            name = 'normal', 
            x = normal_Xscores.index, 
            y = normal_Xscores['score'], 
            mode = 'markers',
            marker_color = '#DBAE58',
            #opacity = 0.7, 
            marker_symbol = 'square'
        )
    )

    outlier_Xscores = score_ts[score_ts['score'] >= threshold]

    fig.add_trace(
        go.Scatter(
            name = 'outlier', 
            x = outlier_Xscores.index, 
            y = outlier_Xscores['score'], 
            mode = 'markers',
            marker_color = '#731331', 
            #opacity = 0.7, 
            marker_symbol = 'square'
        )
    )

    fig.update_yaxes(showgrid = False)
    fig.update_xaxes(showgrid = False)
    fig.update_layout(
        title = '{} : Anomaly scores with a threshold value of {}'.format(model_name, np.round(threshold, 3)), 
        legend=dict( orientation="h", 
                    yanchor="bottom", y=-0.2,
                    xanchor="center", x=0.5
                    )
        )
    if show_image: 
        fig.show()
       
    if path is not None: 
        try: 
            fig.write_image(os.path.join(path, 'anomaly_score_{}_{}.png'.format(model_name, dataname)))
            fig.write_html(os.path.join(path, 'anomaly_score_{}_{}.html'.format(model_name, dataname)))
        except: 
            warnings.warn('Plot cannot be saved')