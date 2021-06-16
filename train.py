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
import tqdm
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
import scanpy as sc
import anndata
import tensorflow as tf
import tensorflow.keras as keras
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin

# local functions
from STANN.models import STANN, BaseSupervisedPCA
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

ap.add_argument(
    "-pcs",
    "--pca_components",
    type=int,
    default=500,
    help="Number of components for sPCA"
)

ap.add_argument(
    "-top",
    "--top_features",
    type=int,
    default=7000,
    help="Number of top features to select"
)

args = vars(ap.parse_args())


################PREPARE OUTPUT DIRS###################
if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

################LOAD DATA###################

# check to see which data
print("[INFO] loading training data...")
adata_train = sc.read_h5ad(args["data_train"])

print("[INFO] loading predict data...")
adata_predict = sc.read_h5ad(args["data_predict"])



################TRAIN TEST SPLIT###################


X_train, Y_train, X_predict = utils.organize_data(adata_train=adata_train,
                                            adata_predict=adata_predict)


X_train_scaled , scaler_train = utils.min_max(X=X_train)

X_predict_scaled , scaler_predict = utils.min_max(X=X_predict)

Y_train_dummy,Y_train_ohe,encoder = utils.label_encoder(Y_train=Y_train)

x_train, x_test, y_train, y_test = utils.get_train_test_split(X_train_scaled,
                                                    Y_train_ohe,
                                                    test_size=0.10, 
                                                    random_state=40)
print(Y_train)                                                   
print(Y_train_ohe)
print(y_train)
class_weights = utils.get_class_weights(Y_train_ohe=y_train)
class_weights = {i : class_weights[i] for i in range(len(Y_train.unique()))}


################RUN SUPERVISED PCA###################

bspca = None
bspca = BaseSupervisedPCA(model=LogisticRegression(multi_class="multinomial",
                                              class_weight=class_weights,
                                              solver='lbfgs'),
                                              n_components=args["pca_components"])

_scores,_scores_balanced = bspca.rank_features(np.array(X_train),
                                                             Y_train_dummy,
                                                             class_weights)
X = bspca.subset_features(x_train,
                 _scores_balanced,
                 args["top_features"])


bspca.fit(X,np.argmax(y_train, axis=1))


################GET SUPERVISED PCA TRANSFORMATIONS###################

x_train_transformed = bspca.get_transformed_data(X)

x_test_subset = bspca.subset_features(x_test,
                 _scores_balanced,
                 args["top_features"])

x_test_transformed = bspca.get_transformed_data(x_test_subset)

X_predict_subset = bspca.subset_features(X_predict.to_numpy(),
                 _scores_balanced,
                 args["top_features"])

X_predict_transformed = bspca.get_transformed_data(X_predict_subset)

################RUN STANN###################

model = STANN(act_fun='tanh',
              first_dense=160,
              second_dense=145.0,
              learning_rate=0.01,input_dim=x_train_transformed.shape[1],
              output_dim=len(adata_train.obs.celltype.unique()))


es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                      mode='min', 
                                      verbose=1,
                                      patience=30)

history = model.fit(x_train_transformed, 
                    y_train, 
                    validation_data=(x_test_transformed, y_test),
                    epochs=30,
                    class_weight=class_weights,
                    callbacks=[es],verbose=0)

utils.print_metrics(model=model,
                  x_train=x_train,
                  y_train=y_train,
                  x_test=x_test,
                  y_test=y_test)


predictions = utils.make_predictions(model=model,
                     X_predict=X_predict_transformed,
                     encoder=encoder,
                     adata_predict=adata_predict,
                     probabilities=False,
                     save=False)


################ SAVE PREDICTIONS ###################

predictions.to_csv(str(args["output"])+str(args["project"])+"_predictions.csv")
_scores_balanced_pd.to_csv(str(args["output"])+str(args["project"])+'_features_scores_balanced.csv')
_scores_pd.to_csv(str(args["output"])+str(args["project"])+'_features_scores.csv')

################ SAVE MODEL ###################

print("[INFO] saving .h5 model ...")
model.save(str(args["output"])+str(args["project"])+"_model.h5")

