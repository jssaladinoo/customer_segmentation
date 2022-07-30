#!/usr/bin/env python
# coding: utf-8

# # Preprocessing module 

import os
import pandas as pd
import numpy as np 

import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import logging, joblib
logger = logging.getLogger(__name__)

class clustering(): 
    def __init__(self, data, algo = 'kmedoids_euclidean', n_clusters = 3): 
        self.data = data
        self.algo = algo
        self.n_clusters = n_clusters
        
        logger.info('Training {}'.format(self.algo))
        
        if self.algo == 'kmedoids_euclidean': 
            logger.info('Training kmediods using euclidean distance.')
            self.model = KMedoids(n_clusters = self.n_clusters, 
                                  metric = 'euclidean', 
                                  random_state = 5
                                 )
        elif self.algo == 'kmedoids_cosine': 
            logger.info('Training kmediods using cosine distance.')
            self.model = KMedoids(n_clusters = self.n_clusters, 
                                  metric = 'cosine', 
                                  random_state = 5
                                 )
        elif self.algo == 'kmedoids_correlation': 
            logger.info('Training kmediods using correlation.')
            self.model = KMedoids(n_clusters = self.n_clusters, 
                                  metric = 'correlation', 
                                  random_state = 5
                                 )
            
        elif self.algo == 'kmeans': 
            logger.info('Training kmeans.')
            self.model = KMeans(n_clusters = self.n_clusters, 
                                  random_state = 5
                                 )
            
        elif self.algo == 'agglomerative_euclidean': 
            logger.info('Training agglomerative cllustering using euclidean distance.')
            self.model = AgglomerativeClustering(
                                n_clusters = self.n_clusters, 
                                affinity = 'euclidean', 
                                linkage = 'ward', 
                                compute_distances = True
                                )
        elif self.algo == 'agglomerative_cosine': 
            logger.info('Training agglomerative cllustering using cosine distance.')
            self.model = AgglomerativeClustering(
                                n_clusters = self.n_clusters, 
                                affinity = 'cosine', 
                                linkage = 'average', 
                                compute_distances = True
                                )
        elif self.algo == 'agglomerative_manhattan': 
            logger.info('Training agglomerative cllustering using manhattan metrics.')
            self.model = AgglomerativeClustering(
                                n_clusters = self.n_clusters, 
                                affinity = 'manhattan', 
                                linkage = 'average', 
                                compute_distances = True
                                )
        elif self.algo == 'gmm_full': 
            logger.info('Training GMM using full covariance type.')
            self.model = GaussianMixture(
                                n_components = self.n_clusters, 
                                covariance_type = 'full', 
                                random_state = 5, 
                                n_init = 5,
                                init_params = 'k-means++'
                                )
        elif self.algo == 'gmm_diag': 
            logger.info('Training GMM using diag covariance type.')
            self.model = GaussianMixture(
                                n_components = self.n_clusters, 
                                covariance_type = 'diag', 
                                random_state = 5, 
                                n_init = 5,
                                init_params = 'k-means++'
                                )

        elif self.algo == 'gmm_spherical': 
            logger.info('Training GMM using spherical covariance type.')
            self.model = GaussianMixture(
                                n_components = self.n_clusters, 
                                covariance_type = 'spherical', 
                                random_state = 5, 
                                n_init = 5,
                                init_params = 'k-means++'
                                )

        elif self.algo == 'gmm_tied': 
            logger.info('Training GMM using tied covariance type.')
            self.model = GaussianMixture(
                                n_components = self.n_clusters, 
                                covariance_type = 'tied', 
                                random_state = 5, 
                                n_init = 5,
                                init_params = 'k-means++'
                                )
        else: 
            logger.info('Model not recognized')
            
    def fit(self): 
        
        if self.algo.startswith('gmm'): 
            self.model.fit(self.data)
            self.labels = self.model.predict(self.data)
            self.score = silhouette_score(self.data, self.labels)
            
        else: 
            self.model.fit(self.data)
            self.labels = self.model.labels_
            self.score = silhouette_score(self.data, self.labels)
        
        
        logger.info('With {} clusters the silhouette score is {}'.format(self.n_clusters, self.score))
        return self
    
    
algo_list = ['kmedoids_euclidean', 'kmedoids_cosine', 
            'kmedoids_correlation', 'kmeans', 
             'agglomerative_euclidean', 'agglomerative_cosine', 
            'gmm_full', 'gmm_diag', 'gmm_tied', 'gmm_spherical' ]

def objective(trial, data): 
    
    tune_params = {
        'n_clusters' : trial.suggest_int('n_clusters', 3, 10), 
        'algo' : trial.suggest_categorical('algo', algo_list)
        }
    
    cls = clustering(data, 
                n_clusters = tune_params['n_clusters'],
                algo = tune_params['algo']
                )
    cls = cls.fit()
    
    return cls.score

def tuner(data, 
          dataname = None, 
          n_trials = 50,
          n_jobs =-1
         ): 
    
    assert dataname != None

    
    search_space = {"n_clusters": list(range(3, 8)), "algo": algo_list}
    study = optuna.create_study(
                    sampler=optuna.samplers.GridSampler(search_space),
                    direction='maximize', 
                    study_name = dataname, 
                    storage='sqlite:///{}.db'.format(dataname), 
                    load_if_exists=True
                    )
    
    study.optimize( lambda trial: objective(trial, data), 
                           n_trials= n_trials, 
                           n_jobs =n_jobs, 
                           gc_after_trial = True
                    )
    
    
    return study
    