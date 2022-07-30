#!/usr/bin/env python
# coding: utf-8

# # Preprocessing module 

import os
import pandas as pd
import numpy as np 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from jenkspy import JenksNaturalBreaks


import logging, joblib
logger = logging.getLogger(__name__)



def fill_na(data): 
    data_ = data.copy()
    missing_columns = data_.columns[data_.isnull().any()]

    for col in missing_columns: 
        data_[col] = data_[col].fillna(data_[col].median())
        
    return data_



def scale_data(data, method = 'standard', return_scaler = False):
    """
    Scale the input data
    """
    if method == 'standard': 
        scaler = StandardScaler()
        if return_scaler: 
            return scaler.fit_transform(data), scaler
        else: 
            return scaler.fit_transform(data)
        
    elif method == 'minmax': 
        scaler = MinMaxScaler()
        if return_scaler: 
            return scaler.fit_transform(data), scaler
        else: 
            return scaler.fit_transform(data)


class Autoencoder(): 
    
    def __init__(self, data, n_layers): 
        self.data = data
        self.n_layers = n_layers
    
        self.n_features = self.data.shape[1]
        if int(self.n_features*(1/(self.n_layers + 2))) < 2: 
            self.n_layers = self.n_layers -1
            
        self.hidden_units = [int(self.n_features*(1/i)) for i in range(1, self.n_layers + 2)]
        self.hidden_units = self.hidden_units + self.hidden_units[:-1][::-1]

        self.autoencoder = MLPRegressor(alpha = 1e-5, 
                                   hidden_layer_sizes=self.hidden_units, 
                                   random_state=5,
                                   max_iter=1000)
        self.autoencoder.fit(self.data, self.data)

        self.weights = self.autoencoder.coefs_
        self.biases = self.autoencoder.intercepts_

        self.encoder_weights = self.weights[0:self.n_layers + 1]
        self.encoder_biases = self.biases[0:self.n_layers + 1]
        
    def encode(self):
        self.reduced_data = self.data.copy()
        
        for index, (w,b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            if index+1 == len(self.encoder_weights):
                self.reduced_data = self.reduced_data@w+b
            else: 
                self.reduced_data = np.maximum(0, self.reduced_data@w+b)
        return self



def reduce_dimension(data, linear = True, dim = 50, 
                    n_layers = 3): 
    """
    This function automatically reduce the dimension of the input data. 
    If linear is True, either PCA or SVD is used. To do this, value for dim should be provided
        If the data is dense, PCA is used. 
        If the data is sparse, SVD is used.
    If linear is False, autoencoder is used. To use this, n_layers should be provided
    
    """
    dim = int(dim)
    n_features = data.shape[1]
    
    if linear: 
        if n_features > dim :     
            # check sparsity
            sparsity = (data == 0).sum().sum()/ data.size

            if sparsity > 0.50: 
                svd = TruncatedSVD(n_components=dim, random_state=5)
                reduced_data = svd.fit_transform(data)
            else: 
                pca = PCA(n_components=dim, random_state=5)
                reduced_data = pca.fit_transform(data)
        else: 
            reduced_data = data
            
        try: 
            return reduced_data
        except: 
            logger.info('Dimensionality reduction failed. Returning original data.')
            return data
    else: 
        ae = Autoencoder(data, n_layers)
        ae = ae.encode()
    
        try: 
            return ae.reduced_data
        except: 
            logger.info('Dimensionality reduction failed. Returning original data.')
            return data


def remove_collinearity(data, threshold = 0.5): 
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features 
    return data.drop(to_drop, axis=1)



def goodness_of_variance_fit(x, groups_):
        x = np.array(x)
        array_mean = np.mean(x)
        sdam = sum([(value - array_mean)**2 for value in x])
        sdcm = 0
        for group in groups_:
            group_mean = np.mean(group)
            sdcm += sum([(value - group_mean)**2 for value in group])
        gvf = (sdam - sdcm)/sdam
        return gvf
    
    
def threshold_jenks(data): 
    data = data[~np.isnan(data)]
    if len(data.shape) <= 1:
        data = np.sort(data).reshape(-1, 1)
        
    nb_class = 4

    gvf = 0
    n_iter = 1
    
    while ((n_iter < 10) and (gvf < 0.95)): 
        try: 
            jnb = JenksNaturalBreaks(nb_class)
            jnb.fit(data)
            gvf = goodness_of_variance_fit(data, groups_= jnb.groups_) 
            nb_class +=1
            n_iter +=1
        except: 
            raise Exception("Jenks failed")
    
    if len(jnb.groups_[-1]) > 1: 
        cut_off_point = np.where(data == jnb.groups_[-1][0])[0]
        cut_off_value = jnb.groups_[-1][0]
    else: 
        cut_off_point = np.where(data == jnb.groups_[-2][0])[0]
        cut_off_value = jnb.groups_[-2][0]
    
    return cut_off_point, cut_off_value, jnb.groups_

class outlier_detection(): 
    
    def __init__(self, data): 
        self.data = data
        
        self.model = IsolationForest(
                        n_estimators = 200,
                        max_samples=min(len(self.data), 100), 
                        random_state = 5, 
                        max_features = .9,
                        contamination= 'auto', 
                        n_jobs = -1
                        )
        
        self.model.fit(self.data)
        self.score = 1 - self.model.decision_function(self.data)
        
        _, self.threshold, self.jenks_group = threshold_jenks(data=self.score)
        
        self.outlier_indicator = (self.score > self.threshold)*1
        self.outlier_idx = list(np.where(self.outlier_indicator== 1)[0])
        self.non_outlier_idx = list(np.where(self.outlier_indicator== 0)[0])


        self.score_ = pd.DataFrame(self.score, columns = ['score'])
