import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import torch
#import torchvision
#import torch.nn as nn
#import torch.optim as optim
#import torchvision.transforms as T
import scipy
import math
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('--load-dir', 
    help='Name of classifier to use for oracle anomaly detector.', 
    default='random_forest')

args = parser.parse_args()

director = args.load_dir

X_in_Otrain = np.load(os.path.join(director, "val_zs.npy"))
Y_in_Otrain = np.array([0] * ( X_in_Otrain.shape[0]))

X_test_inliers = np.load(os.path.join(director, "test_zs.npy"))
Y_test_inliers = np.array([0] * ( X_test_inliers.shape[0]))


X_outliers = np.load(os.path.join(director, "open_zs.npy"))
outlier_split = int(X_outliers.shape[0]/10)

X_out_Otrain = X_outliers[:outlier_split]
X_out_test   = X_outliers[outlier_split:]

Y_out_Otrain = np.array([1] * ( X_out_Otrain.shape[0]))
Y_out_test   = np.array([1] * ( X_out_test.shape[0]))


eval_x = np.concatenate([X_test_inliers,X_out_test])
eval_y = np.concatenate([Y_test_inliers,Y_out_test])




# Train the oracle
#oracle = MLPClassifier(random_state=0, solver='adam', learning_rate='adaptive', hidden_layer_sizes=(X_in_Otrain.shape[1]*4,X_in_Otrain.shape[1]))
#oracle = RandomForestClassifier(max_depth=10, random_state=0)
oracle = LinearSVC(max_iter=5000,random_state=0)
oracle.fit(np.concatenate([X_in_Otrain, X_out_Otrain]), np.concatenate([Y_in_Otrain, Y_out_Otrain]))
    


''' Get oracle accuracy, AUC, and AUC Curve '''
# Load the latent test examples into memory 

accuracy  = oracle.score(eval_x, eval_y)
#auc_score = roc_auc_score(eval_y, oracle.predict_proba(eval_x)[:, 1])
#fpr, tpr, thresholds = roc_curve(eval_y, oracle.predict_proba(eval_x)[:, 1])
auc_score = roc_auc_score(eval_y, oracle.decision_function(eval_x))
fpr, tpr, thresholds = roc_curve(eval_y, oracle.decision_function(eval_x))
tnrattpr95 = 1 - fpr[np.argmax(tpr>=.95)]

out_str = (",".join(os.path.basename(os.path.normpath(director)).split("_"))).replace(",resize","_resize")

print(f'{out_str},{accuracy},{auc_score},{tnrattpr95}')


