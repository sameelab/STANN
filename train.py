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
Date: 12/2020

"""

# USAGE
# python train.py --model STANN --data seqfish --output AE_seqfish_1

print("[INFO] loading libraries...")
#import the necessary packages
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import pandas as pd
import warnings
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
import scanpy as sc
import anndata
import tensorflow as tf
import tensorflow.keras as keras
import scanpy as sc

# local functions
from STANN.models import STANN
import STANN.utils as utils

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

#Reproducibility
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

################construct the argument parser and parse the arguments###################
ap = argparse.ArgumentParser()

ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="STANN",
    choices=["STANN", "OTHER"],
    help="type of model architecture",
)

ap.add_argument(
    "-dt",
    "--data_train",
    type=str,
    required=True,
    help="Path of training dataset"
)

ap.add_argument(
    "-dp",
    "--data_predict",
    type=str,
    required=True,
    help="Path of prediction dataset"
)

ap.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path of outputs"
)

ap.add_argument(
    "-p",
    "--project",
    type=str,
    required=True,
    help="Path of outputs"
)

ap.add_argument(
    "-cv",
    "--cross_validate",
    type=str,
    default=False,
    choices=[True, False],
    help="cross-validation"
)


args = vars(ap.parse_args())


################LOAD DATA###################

# check to see which data
print("[INFO] loading training data...")
adata_train = sc.read_h5ad(args["data_train"])

print("[INFO] loading predict data...")
adata_predict = sc.read_h5ad(args["data_predict"])

model = STANN(act_fun='tanh',
              first_dense=160,
              second_dense=145.0,
              learning_rate=0.01,input_dim=adata_train.X.shape[1],
              output_dim=len(adata_train.obs.celltype.unique()))

X_train, Y_train, X_predict = utils.organize_data(adata_train=adata_train,
                                            adata_predict=adata_predict)


X_train_scaled , scaler_train = utils.min_max(X=X_train)

X_predict_scaled , scaler_predict = utils.min_max(X=X_predict)

Y_train_dummy,Y_train_ohe,encoder = utils.label_encoder(Y_train=Y_train)

x_train, x_test, y_train, y_test = utils.get_train_test_split(X_train_scaled,
                                                    Y_train_ohe,
                                                    test_size=0.10, 
                                                    random_state=40)

class_weights = utils.get_class_weights(Y_train_ohe=y_train)
class_weights = {i : class_weights[i] for i in range(15)}

es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                      mode='min', 
                                      verbose=1,
                                      patience=30)

history = model.fit(x_train, 
                    y_train, 
                    validation_data=(x_test, y_test),
                    epochs=30,
                    class_weight=class_weights,
                    callbacks=[es],verbose=0)

utils.print_metrics(model=model,
                  x_train=x_train,
                  y_train=y_train,
                  x_test=x_test,
                  y_test=y_test)

predictions = utils.make_predictions(model=model,
                     X_predict=X_predict_scaled,
                     encoder=encoder,
                     adata_predict=adata_predict,
                     probabilities=False,
                     save=False)


################ SAVE PREDICTIONS ###################

if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

predictions.to_csv(str(args["output"])+str(args["project"])+"_predictions.csv")

################ SAVE MODEL ###################

print("[INFO] saving .h5 model ...")
model.save(str(args["output"])+str(args["project"])+"_model.h5")

