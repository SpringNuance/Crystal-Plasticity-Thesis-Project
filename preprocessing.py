import pandas as pd
import numpy as np
import os 
import glob
import csv
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import os
os.getcwd()

## Data preprocessing
def preprocess(path):
    df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    trueStress = df["Mises(Cauchy)"]
    trueStrain = df["Mises(ln(V))"]
    truePlasticStrain = getTruePlasticStrain(trueStress, trueStrain)
    return trueStress, trueStrain, truePlasticStrain

## Data preprocessing
def preprocessExperimentalCurve(path, usecols):
    df = pd.read_excel(path , skiprows= 7, nrows= 174, usecols = usecols)
    trueStress = df.iloc[:, 0]
    trueStrain = df.iloc[:, 1]
    truePlasticStrain = getTruePlasticStrain(trueStress, trueStrain)
    return trueStress, trueStrain, truePlasticStrain

def getTruePlasticStrain(trueStress, trueStrain):
    # numpy therefore index is 0 
    # Getting the slope
    # truePlasticStrain = trueStrain - trueElasticstrain = trueStrain - trueStress/Young's modulus
    Young = (trueStress[1] - trueStress[0]) / (trueStrain[1] - trueStrain[0])
    truePlasticStrain = trueStrain - trueStress / Young    
    return truePlasticStrain

def elastic(trueStress, trueStrain, truePlasticStrain):
    for val in range(trueStrain.size, 5, -1):
        elasticStrain = trueStrain[0:val]
        elasticStress = trueStress[0:val]
        r2 = adjR(elasticStrain, elasticStress, 1)
        #print(val, r2)
        if r2 > 0.998:
            break
    trimmedStrain = trueStrain[val - 1:trueStrain.size]
    trimmedStress = trueStress[val - 1:trueStress.size] 
    return elasticStress, elasticStrain, trimmedStress, trimmedStrain, r2

def plot(elasticStress, elasticStrain, trimmedStress, trimmedStrain, trueStress, truePlasticStrain, r2):
    plt.plot(trimmedStrain, trimmedStress, 'g', label = "Plastic True Stress - True Strain")
    plt.plot(truePlasticStrain, trueStress, 'b', label = "Flow Curve")
    plt.plot(elasticStrain, elasticStress, 'r', label = "Elastic True Stress - True Strain")
    leg = plt.legend(loc = "upper right")
    plt.xlabel(xlabel = "Strain (mm)")
    plt.ylabel(ylabel = "Stress (MPa)")
    plt.xlim([0, 0.005])
    plt.figure(figsize = (6,6))
    plt.show()
    print("Adjusted R squared = ", str(r2))

def adjR(x, y, degree):
    results = []
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))
    return results
    
def plotTrue(trueStress, trueStrain, truePlasticStrain):
    plt.plot(trueStrain, trueStress, 'g', label = "trueStress - trueStrain")
    plt.plot(truePlasticStrain, trueStress, 'b', label = "Flow curve: trueStress - truePlasticStrain")
    leg = plt.legend(loc = "upper right")
    plt.xlabel(xlabel = "Strain (mm)")
    plt.ylabel(ylabel = "Stress (MPa)")
    plt.xlim([0, 0.005])
    plt.figure(figsize = (6,6))
    plt.show()