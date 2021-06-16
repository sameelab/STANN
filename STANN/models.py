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
from numpy.random import seed
seed(123)
#from tensorflow import set_random_seed
#set_random_seed(234)

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
