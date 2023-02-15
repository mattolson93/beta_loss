import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import scipy
import math
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def load_Xy(ZX_filename, Zy_filename):
    ''' Loads example and lable data from specified .npy files. '''   
    with open(ZX_filename, 'rb') as f:
        X = np.load(f)

    with open(Zy_filename, 'rb') as f:
        y = np.load(f)

    return X, y


def train_oracle(args):
    ''' Train the oracle anomaly detector. '''
 
    # Load the latent training examples into memory
    ZTrainX, ZTrainy = load_Xy(args.trainX_filename, args.trainy_filename)
    
    # TEST
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)

    # Train the oracle
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    #clf.fit(ZTrainX, ZTrainy)
    
    clf.fit(X,y)
    #print('Prediction')
    #print(clf.predict([[0, 0, 0, 0]]))
    
    return clf


def query_oracle(oracle, args):
    ''' Get oracle accuracy, AUC, and AUC Curve '''
         
    # Load the latent test examples into memory 
    ZTestX,  ZTesty  = load_Xy(args.testX_filename, args.testy_filename)
  
      
    # TEST
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
 
    accuracy  = oracle.score(X, y)
    auc_score = roc_auc_score(y, oracle.predict_proba(X)[:, 1])
    fpr, tpr, thresholds = roc_curve(y, oracle.predict_proba(X)[:, 1])

    #accuracy  = oracle.score(ZTestX, ZTesty)
    #auc_score = roc_auc_score(ZTesty, oracle.predict_proba(ZTestX)[:, 1])
    #fpr, tpr, thresholds = roc_curve(ZTesty, oracle.predict_proba(ZTestX)[:, 1])
    
    return {'accuracy': accuracy, 'auc': auc_score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}


def write_results(results, args):
    ''' Writes the oracle's evaluation to disk. '''
    output_filename = '{}_oracle_out_{}_loss_{}.txt'.format(args.classifier, args.loss_fn_name, args.dataset_name)
    with open(output_filename, 'a+') as f:
        f.write('\n===== Results for {} loss experiment on dataset {}: =====\n'.format(args.loss_fn_name, args.dataset_name))
        f.write('Accuracy:  {}\n\n'.format(results['accuracy']))
        f.write('AUC:       {}\n\n'.format(results['auc']))
        f.write('fpr:       {}\n\n'.format(results['fpr']))
        f.write('tpr:       {}\n\n'.format(results['tpr']))
        f.write('thresholds {}\n\n'.format(results['thresholds']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', 
        help='Name of classifier to use for oracle anomaly detector.', 
        default='random_forest')
    parser.add_argument('--dataset_name', 
        help='Name of the dataset used to generate the embeddings. [default: None]')
    parser.add_argument('--loss_fn_name',
        help='Name of the loss function used to train the model that produced the embedded data. [default: None]')
    parser.add_argument('--trainX_filename', 
        help='File containing embedded training examples. [default: ZTrainX.npy]',
        default='ZTrainX.npy')
    parser.add_argument('--trainy_filename',
        help='Name of file containing embedded training example labels. [default: ZTrainy.npy]',
        default='ZTrainy.npy')
    parser.add_argument('--testX_filename', 
        help='Name of file containing embedded test examples. [default: ZTestX.npy]',
        default='ZTestX.npy')
    parser.add_argument('--testy_filename',
        help='Name of file containing embedded test example labels. [default: ZTesty.npy]',
        default='ZTesty.npy')

    args = parser.parse_args()

    # FOR TEST
    x = np.random.randn(5,3)
    y = np.random.randn(5,1)
    xt = np.random.randn(6,5)
    yt = np.random.randn(6,1)
    with open('ZTrainX.npy', 'wb') as f:
        np.save(f, x)
    
    with open('ZTrainy.npy', 'wb') as f:
        np.save(f, y)
    
    with open('ZTestX.npy', 'wb') as f:
        np.save(f, xt)
    
    with open('ZTesty.npy', 'wb') as f:
        np.save(f, yt)

    oracle  = train_oracle(args)
    results = query_oracle(oracle, args)
    write_results(results, args)  
  

if __name__ == '__main__':
    main()	
