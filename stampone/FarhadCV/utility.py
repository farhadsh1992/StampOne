#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:16:41 2019

@author: Farhad

INFO: Misc training/data-prep helpers: CPU/ETA progress logging and
pandas dummy/label-encoding utilities for tabular model inputs.
"""

import pendulum
import psutil
import os 
import sys

import base64
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing

def estimate_cpu(num,start,lenghtdf):
    numb=num+1
    stop = pendulum.now()
    dift1 = abs(start-stop)*1
    estimatTime = abs(abs(dift1*lenghtdf/numb)-dift1)
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = psutil.virtual_memory().percent
    cpuUse = psutil.cpu_percent(interval=1, percpu=True)
     #aa+=1
    run = ("["+str(num+1)+
           '/'+str(lenghtdf)+"] _ [ time:" + str(dift1) +
           "] _ ["+" Remain:" + str(estimatTime)+
           "] _ ["+" Memory:"+str(memoryUse)+"% ] _ ["+" CPU:"+str(cpuUse)+"% ]")
    sys.stdout.write('\r'+ run)
    
    
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy_array(Array):
    df = pd.DataFrame({'label_':Array},dtype=float)
    name = 'label_'
    dummies = pd.get_dummies(df[name],dtype=float)
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    return df





# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy_df(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = f"{name}-{tv}"
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

def report_coef(names,coef,intercept):
    r = pd.DataFrame( { 'coef': coef, 'positive': coef>=0  }, index = names )
    r = r.sort_values(by=['coef'])
    r2 = r.style.applymap(color_negative_red)
    display(r2)
    print("Intercept: {}".format(intercept))
    r['coef'].plot(kind='barh', color=r['positive'].map({True: 'b', False: 'r'}))
    plt.title('Regression coefficients')
    plt.show()