#!/usr/bin/env python
# coding: utf-8

# Required packages


import numpy as np
import pandas as pd
import os, sys
from IPython.display import display
from pycaret.utils import version


# Getting the data

train = pd.read_csv('Il_10_AAC_SMOTE.csv')


# Setting up Environment in PyCaret


from pycaret.classification import *

clf1 = setup(
    data=train,
    target='Class',
    train_size=0.80,
    feature_selection=True,
    feature_selection_threshold=0.9,
    feature_selection_method='classic',
    fold=5,
    data_split_stratify=True,
    session_id=123,
    log_experiment=True,
    experiment_name='il-10_transformd_Azure'
)

#Comparing All Models

best = compare_models()

#Create a Model 'lightgbm'

lightgbm = create_model('lightgbm')

#trained model object is stored in the variable:"lightgbm"

print(lightgbm)

#Plot a Model
plot_model(estimator = lightgbm)#AUC
plot_model(estimator = lightgbm, plot = 'confusion_matrix')
plot_model(estimator = lightgbm, plot = 'feature')
#Predict on test / hold-out Sample
predict_model(lightgbm)



#Create a Model
et = create_model('et')
#trained model object is stored in the variable:"et"
print(et)

# Plot Model
plot_model(estimator = et)
plot_model(estimator = et, plot = 'confusion_matrix')
plot_model(estimator = tuned_et, plot = 'feature')

#Predict on test / hold-out Sample
predict_model(et)


#Create a Model
catboost = create_model('catboost')

# trained model object is stored in the variable:"catboost"

print(catboost)

# Plot Model

plot_model(estimator = catboost)
plot_model(estimator = catboost, plot = 'confusion_matrix')
plot_model(estimator = catboost, plot = 'feature')


#Predict on test / hold-out Sample
predict_model(catboost)
