"""
   ___                  _                
  / _/______ ____  ____(_)__ _______     
 / _/ __/ _ `/ _ \/ __/ (_-</ __/ _ \    
/_//_/  \_,_/_//_/\__/_/___/\__/\___/    
  ___ _____(_)__ ___ ____  / /_(_)       
 / _ `/ __/ (_-</ _ `/ _ \/ __/ /        
 \_, /_/ /_/___/\_,_/_//_/\__/_/         
/___/


Samee Lab @ Baylor College Of Medicine
francisco.grisanticanozo@bcm.edu
Date: 12/2019

"""


import numpy as np
import os
import sys
import tqdm
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin


#Reproducibility
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

def STANN(act_fun='relu',
                 first_dense=200,
                 second_dense=155,
                 learning_rate=0.001,
                 input_dim=None,
                 output_dim=None):  
    """
    initiates and creates keras sequential model
    returns -> model
    """
    model = None
    
    model = keras.Sequential()
    
    model.add(tf.keras.layers.Dense(first_dense, 
                    input_dim=input_dim,
                    activation=act_fun,kernel_initializer='he_uniform'))
    
    model.add(tf.keras.layers.Dense(second_dense, 
                    input_dim=input_dim,
                    activation=act_fun))
    
    model.add(tf.keras.layers.Dense(output_dim, 
                    activation='softmax')) #21 diff clusters
    
    #compile the keras model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate), 
                  metrics=['accuracy'])
    
    return model


  
class BaseSupervisedPCA(object):
    """
    Supervised PCA algorithm proposed by Bair et al. (2006).
    
    
    Parameters
    ----------
    
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        
    model : The supervised learning model that will be used to conduct supervised PCA.
    
    Attributes
    ----------
        
    
    References
    ----------
    Bair, Eric, et al. "Prediction by supervised principal components." Journal of the American Statistical Association 101.473 (2006).
    
    """
    
    def __init__(self, 
                 fit_intercept=True, 
                 model=None,
                 threshold=0.5,
                 n_components=-1):
        
        
        self.fit_intercept = fit_intercept
        self._model=model
        self._pca=None
        self._leavouts=None
        self._scores=None
        self._scores_balanced=None
        
        self._threshold=threshold
        self._n_components=n_components
    
    
    def rank_features(self,X,y,weights):
        
        #these are the columns that will be removed
        self._leavouts=[]        
        self._scores=[] 
        self._scores_balanced=[]
        
        dummy_X=X[:,np.newaxis]
        
        #test the absolute value of the coefficient for each variable. If it
        #is below a the threshold, then add it to the leaveouts 
        
        
        print('[INFO] Running feature selection')
        
        for i in tqdm.tqdm(range(0,dummy_X.shape[2]),total=dummy_X.shape[2]):
            
            current_X=dummy_X[:,:,i]
            self._model.fit(current_X, y)
            #the all([]) syntax is there in order to support both linear and logistic
            #regression. Logistic regression coefficients for multiclass problems
            #come in multi-dimensional arrays.
            #print(self._model.coef_)
            
            self._model.predict(current_X)
            
            #print(current_X.shape)
            score_balanced = sklearn.metrics.balanced_accuracy_score(y, 
                                                    self._model.predict(current_X),
                                                    #sample_weight=weights, 
                                                    adjusted=False)
            
            score_not_balanced = sklearn.metrics.accuracy_score(y, self._model.predict(current_X))
            
            self._scores.append(score_not_balanced)
            self._scores_balanced.append(score_balanced)
        
        
        return self._scores, self._scores_balanced
    
    def subset_features(self,
                        X,
                        _scores,
                        top=5000):

        
        
        dummy_X=X[:,np.newaxis]
        #delete the variables that were below the threshold
        if(len(_leaveouts)>0):
            
            dummy_X=np.delete(dummy_X,_leaveouts,2)
            
        
            
        if(len(_leaveouts)==dummy_X.shape[2]):
            raise ValueError('The total number of features to be left out is equal to the total number of features. Please try with a smaller threshold value.')
            
        
        return dummy_X
        

        
    #def subset_feature(self,X,y,)
    #
    #    #delete the variables that were below the threshold
    #    if(len(self._leavouts)>0):
    #        dummy_X=np.delete(dummy_X,self._leavouts,2)
    #    
    #    return 
    
    def fit(self,X,y):
        """
        Fit the supervised PCA model
        .
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        threshold : the threshold for the coefficient below which it is discarded.
        n_components : the number of components to keep, after running PCA
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        
        #conduct PCA for the designated number of components.
        #If no number was designated (or an illegal value<=0) then use the max number of component
        
        if(self._n_components>0):
            self._pca = PCA(n_components=self._n_components)
        else:
            self._pca = PCA(n_components=X.shape[2])
            
        dummy_X=self._pca.fit_transform(X[:,0,:])
        
        self._model=self._model.fit(dummy_X,y)
        
        
        return self
        
    def predict(self,X):
        """Predict using the supervised PCA model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.        
        """
        #remove the leavouts, transform the data and fit the regression model
        transformed_X=self.get_transformed_data(X)
        return self._model.predict(transformed_X)
    
    def get_transformed_data(self,X):
        """Calculates the components on a new matrix.
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            
        Returns
        -------
        transformed_X: Returns a transformed numpy array or sparse matrix. The
        leavouts have been removed and the remaining variables are transformed into
        components using the weights of the PCA object.
        
        Notes
        -------
        The algorithm should have first been executed on a dataset.
        
        """
        #transformed_X=np.delete(X,self._leavouts,1)
        
        transformed_X = X[:,0,:]
        transformed_X=self._pca.transform(transformed_X)
        
        return transformed_X
        
    def get_n_components(self):
        return self._pca.n_components_
    
    
    #I am implementing a function here to get the components in order to avoid
    #the user having to access the pca object. Another option would be to 
    #copy the components from the pca to a variable located at 'self'. However,
    #this might be too redundant.
    
    def get_components(self):
        """Returns the components formerly calculated on a training dataset.
            
        Returns
        -------
        components: A numpy matrix with the loadings of the PCA components.
        
        Notes
        -------
        The algorithm should have first been executed on a dataset.
        
        """
        return self._pca.components_
    
    #same principle as in the get_components function
    def get_coefs(self):
        return self._model.coef_
        
    def score(self,X,y):
        return self._model.score(X,y)