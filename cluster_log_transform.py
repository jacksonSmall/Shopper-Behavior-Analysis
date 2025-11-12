# Cluster_Log Transform Class Script
# Used for preprocessing to transform dataset from raw state to make it ready for training

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Cluster_Log(BaseEstimator,TransformerMixin):
    def __init__(self, k=5):
        self.k = k
        self.model = KMeans(n_clusters=self.k,
                            random_state=0,
                            max_iter=2000)
        self.scaler = MinMaxScaler()   
    
    # Fits KMeans clusters based on Min-Max scaled data
    def fit(self, X, y=None):
        self.scaled = self.scaler.fit_transform(X)
        self.model.fit(self.scaled)
        return self
    
    # Adds cluster labels, log-transformed product-related features, and drops the original product-related features
    def transform(self, X, y=None):
        self.scaled = self.scaler.transform(X)
        labels = self.model.predict(self.scaled)
        return pd.concat(
            [X.drop(columns=['ProductRelated','ProductRelated_Duration']).reset_index(drop=True),
             pd.get_dummies(labels, prefix='cluster', dtype='int').reset_index(drop=True),
             np.log1p(X[['ProductRelated','ProductRelated_Duration']]).reset_index(drop=True)], ignore_index=True, axis=1)