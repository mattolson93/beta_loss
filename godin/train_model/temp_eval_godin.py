# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import argparse
import os

from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from deconfnet import DeconfNet, NormalLinear, GodinLayer, RatioLayer, SoftplusLinear, GodinLayerSoftplus
from datasets import OpenDatasets


from tqdm import tqdm

r_mean = 125.3/255
g_mean = 123.0/255
b_mean = 113.9/255
r_std = 63.0/255
g_std = 62.1/255
b_std = 66.7/255

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])




class losses:
    def __init__(self, do_softplus=False):
        self.softplus = None#nn.Softplus() if do_softplus else lambda x: x


    def kliep_loss(self, logits, labels, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        

        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def kliep_loss_sigmoid(self, logits, labels, max_ratio=10):
        preds  = torch.sigmoid(logits) * 10

        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def ulsif_loss(self, logits, labels, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (-2*(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds**2)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss


    def power_loss(self, logits, labels, alpha=.1, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (1 - preds.pow(alpha))/(alpha)).sum(1)
        outlier_loss = ((1-labels) * (preds.pow(1+alpha)-1)/(1+alpha)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def power_loss_05(self, logits, labels): return self.power_loss(logits, labels, alpha=.05, max_ratio=50)
    def power_loss_10(self, logits, labels): return self.power_loss(logits, labels, alpha=.1, max_ratio=50)
    def power_loss_50(self, logits, labels): return self.power_loss(logits, labels, alpha=.5, max_ratio=50)
    def power_loss_90(self, logits, labels): return self.power_loss(logits, labels, alpha=.90, max_ratio=50)

    def get_loss_dict(self):
        return {
            'ce':nn.CrossEntropyLoss(),
            'kliep':   self.kliep_loss, 
            'ulsif':   self.ulsif_loss, 
            'power05': self.power_loss_05, 
            'power10': self.power_loss_10, 
            'power50': self.power_loss_50, 
            'power90': self.power_loss_90, 
        }




def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
    # Device arguments
    parser.add_argument('--gpu', default = 0, type = int,
                        help = 'gpu index')

    # Model loading arguments
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--model-dir', default = './outmodels', type = str,
                        help = 'model name for saving')

    # Architecture arguments
    parser.add_argument('--architecture', default = 'densenet', type = str,
                        help = 'underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument('--layer-type', default = 'cosine', type = str,
                        help = 'similarity function for decomposed confidence numerator (cosine | inner | euclid | baseline)')
    parser.add_argument('--loss-type', default = 'ce', type = str,
                        help = 'ce|kliep|power05|power10|power50|power90|ulsif')
    parser.add_argument('--opencat-exp', default = 'None', type = str,
                        help = 'mnist|svhn|cifar10|cifar10+|cifar50+|tinyimagenet')

    # Data loading arguments
    parser.add_argument('--data-dir', default='./data', type = str)
    parser.add_argument('--out-dataset', default = 'Imagenet', type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--batch-size', default = 64, type = int,
                        help = 'batch size')
    parser.add_argument('--do-softplus', default = 1, type = int,
                        help = 'batch size')


    # Training arguments
    parser.add_argument('--no-train', action='store_false', dest='train')
    parser.add_argument('--weight-decay', default = 0.0001, type = float,
                        help = 'weight decay during training')
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'number of epochs during training')
    parser.add_argument('--seed', default = 0, type = int,
                        help = 'number of epochs during training')

    # Testing arguments
    parser.add_argument('--no-test', action='store_false', dest='test')
    parser.add_argument('--magnitudes', nargs = '+', default = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                        help = 'perturbation magnitudes')
    
    
    parser.set_defaults(argument=True)
    return parser.parse_args()

    

def get_opencat(data_dir, oc_name, batch_size, seed=1):
    dataObject = OpenDatasets(batch_size, batch_size, num_workers=4, add_noisy_instances=False)

    opencat_dict =  {
        "mnist":        ("mnist","mnist",6,4),
        "svhn":         ("svhn","svhn",6,4),
        "cifar10":      ("cifar10","cifar10",6,4),
        "cifar10+":     ("cifar10","cifar100",4,10),
        "cifar50+":     ("cifar10","cifar100",4,50),
        "tinyimagenet": ("tinyimagenet","tinyimagenet",20,200),
    }

    inlier_dataset, outlier_dataset, inlier_count, outlier_count = opencat_dict[oc_name]
    

    sectioned_inliers, sectioned_outliers = None, None

    if inlier_dataset == 'cifar10' and outlier_dataset  == 'cifar100':
        if inlier_count != 4 or outlier_count <= 0: exit("cannot run cifar10 vs cifar100 unless inlier is 4 and outlier >0")
        sectioned_inliers, sectioned_outliers = dataObject.get_random_cifar100(seed, outlier_count)
    elif inlier_dataset == outlier_dataset:
        print("using only one dataset ")
        num_total_classes = dataObject.get_dataset_class_count(inlier_dataset)
        if inlier_count >= num_total_classes: raise "inlier_count too high"
        sectioned_inliers, sectioned_outliers = dataObject.set_random_classes(seed, inlier_count, num_total_classes)

    elif outlier_count > 0:
        exit("ARGUMENT 'outlier_count' not implemented excpet for cifar10 vs cifar100")

   

    train_loader   = dataObject.get_dataset_byname(inlier_dataset, train=True, specific_classes=sectioned_inliers )  
    train_loader   = dataObject.test_DataLoader(train_loader.dataset)
    test_loader    = dataObject.get_dataset_byname(inlier_dataset, train=False,specific_classes=sectioned_inliers)  
    
    #hack to get val data
    test_dataset = test_loader.dataset
    test_indices = list(range(len(test_loader.dataset)))
    val_len = int(len(test_loader.dataset)/10)
    val_set_in = Subset(test_dataset, test_indices[:val_len])
    test_set_in       = Subset(test_dataset, test_indices[val_len:])
    
    val_loader  = dataObject.test_DataLoader(val_set_in)
    test_loader = dataObject.test_DataLoader(test_set_in)


    openset_loader = dataObject.get_dataset_byname(outlier_dataset,train=False,specific_classes=sectioned_outliers)  

    return train_loader, val_loader, test_loader, openset_loader, inlier_count

def get_datasets(data_dir, data_name, batch_size):

    train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transform)
    test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transform)
    
    if data_name == 'Gaussian' or data_name == 'Uniform':
        normalizer = Normalizer(r_mean, g_mean, b_mean, r_std, g_std, b_std)
        outlier_loader = generating_loaders_dict[data_name](batch_size = batch_size, num_batches = int(10000 / batch_size), transformers = [normalizer])
    else:
        outlier_set  = torchvision.datasets.ImageFolder(f'{data_dir}/{data_name}', transform=test_transform)
        outlier_loader       =  DataLoader(outlier_set,       batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_indices      = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in       = Subset(test_set_in, test_indices[1000:])

    train_loader_in      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
    validation_loader_in =  DataLoader(validation_set_in, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader


def main():
    args = get_args()
    
    device           = "cuda"

    layer_type       = args.layer_type
    loss_type        = args.loss_type
    
    data_dir         = args.data_dir
    model_dir        = args.model_dir
    data_name        = args.out_dataset
    batch_size       = args.batch_size
    
    weight_decay     = args.weight_decay
    epochs           = args.epochs
    seed             = args.seed
    noise_magnitudes = args.magnitudes

    if layer_type == "linear" and loss_type != 'ce': exit("linear layer can only be used with CrossEntropyLoss")
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.opencat_exp != "None":
        dataset_name = args.opencat_exp
        if dataset_name == "mnist": exit("havent implemented mnist yet for densenet")
        train_data, val_data, test_data, open_data, num_classes = get_opencat(data_dir, args.opencat_exp, batch_size, seed)
    else:
        #get outlier data
        dataset_name = 'cifarall'
        num_classes = 10
        train_data, val_data, test_data, open_data = get_datasets(data_dir, data_name, batch_size)

    # Create necessary directories
    exp_name = f'{loss_type}_{layer_type}_{dataset_name}_{seed}'

    print("running experiment of name: ",exp_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    CustomLayers = {
        'linear': NormalLinear,
        'godin': GodinLayer,
        'godinsp': GodinLayerSoftplus,
        'ratio':RatioLayer,
        'linearsp': SoftplusLinear,
    }

    underlying_net = DenseNet3(depth = 100, num_classes = num_classes).to(device)
    last_layer = CustomLayers[layer_type](underlying_net.output_size, num_classes).to(device)

    criterion = losses()
    criterion = criterion.get_loss_dict()[loss_type]


    outmodel_pth = os.path.join(model_dir,exp_name+'.pth')
    print("loading model from ", outmodel_pth)  
    model = DeconfNet(underlying_net, last_layer).to(device)
    model.load_state_dict(torch.load(outmodel_pth))
    model.eval()


    best_auc = None

    if "godin" in layer_type:
        scoring_func = ["max_h","max_g"]
    elif "linear" in layer_type:
        scoring_func = ["ce_logit","max_logit","norm_logit"]
    elif "ratio" in layer_type:
        scoring_func = ["norm_logit","norm_h","max_logit","latent_h","max_h","max_g","norm_g"]
    else:
        exit("bad layer type")

    if dataset_name == 'cifarall': 
        dataset_name+=data_name
        exp_name = f'{loss_type}_{layer_type}_{dataset_name}_{seed}'


    out_zs_dir = os.path.join("out_models", exp_name)
    score_func = ""
    bestacc = 0
    best_auroc = 0
    not_saved = True
    outstr =""
    #validation_results, val_labs,     val_zs  = testData_getinfo(model, device, val_data  , 0, criterion, score_func, title = 'Validating')
    acc, id_test_results, test_labs,  test_zs = testData_getinfo(model, device, test_data , 0, criterion, score_func, title = 'Testing ID', get_acc=True) 
    ood_test_results,  _ ,            open_zs = testData_getinfo(model, device, open_data , 0, criterion, score_func, title = 'Testing OOD')
    
    np.savetxt("labels.csv",test_labs, delimiter=',')
    np.savetxt("dre_test_ratio.csv",  id_test_results[0], delimiter=',') 
    np.savetxt("dre_test_p_yx.csv",  id_test_results[1], delimiter=',') 
    np.savetxt("dre_test_p_notyx.csv",  id_test_results[2], delimiter=',') 
    np.savetxt("dre_open_ratio.csv", ood_test_results[0], delimiter=',') 
    np.savetxt("dre_open_p_yx.csv", ood_test_results[1], delimiter=',') 
    np.savetxt("dre_open_p_notyx.csv", ood_test_results[2], delimiter=',') 

       



def calc_tnr(id_test_results, ood_test_results, tpr_thresh=.95):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=tpr_thresh)]

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    trues  = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result


def get_score_by_type(model_output, scoretype, z):
    
    if type(model_output) is not tuple:
        pred = model_output
    elif "_h" in scoretype:
        pred = model_output[1]
    elif '_g' in scoretype:
        pred = model_output[2]
    elif '_logit' in scoretype:
        pred = model_output[0]
    else:
        exit("bad score type1")

    if "latent" in scoretype :
        score = torch.norm(z , p=1, dim=1)
    elif 'max' in scoretype:
        score = torch.max(pred, dim = 1)[0]
    elif 'norm' in scoretype:
        score = torch.norm(pred , p=1, dim=1)
    elif 'ce' in scoretype:
        score = torch.max(torch.softmax(pred, dim=1), dim=1)[0]
    else:
        exit("bad score type2")

    return pred, score


def testData(model, CUDA_DEVICE, data_loader, noise_magnitude, criterion, score_func = 'h', title = 'Testing', get_acc=False):
    model.eval()
    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)

    total = 0
    total_correct = 0

    for j, (images, labels) in enumerate(data_iter):
        data_iter.set_description(f'{title} | Processing image batch {j + 1}/{num_batches}')
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        
        
        model_output = model(images)
        pred, score = get_score_by_type(model_output, score_func, model.z)

        if get_acc:
            total += len(images)
            total_correct += torch.sum(pred.max(dim=1)[1].cpu() == labels)
        if noise_magnitude != 0.0:
            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            score.backward(torch.ones(len(score)).to(CUDA_DEVICE))
            
            # Normalizing the gradient to binary in {-1, 1}
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
            gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
            gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)
        
            # Now calculate score
            model_output = model(tempInputs)

            pred, score = get_score_by_type(model_output, score_func, model.z)

        results.extend(score.data.cpu().numpy())
        
    data_iter.set_description(f'{title} | Processing image batch {num_batches}/{num_batches}')
    data_iter.close()

    if get_acc:
        accuracy = float(total_correct) / total
        print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))

    if get_acc:
        return accuracy,  np.array(results)
    return np.array(results)


softmax_scores = False

@torch.no_grad()
def testData_getinfo(model, CUDA_DEVICE, data_loader, noise_magnitude, criterion, score_func = 'h', title = 'Testing', get_acc=False):
    model.eval()
    num_batches = len(data_loader)
    results1 = []
    results2 = []
    results3 = []

    total = 0
    total_correct = 0
    labs = []
    zs = torch.empty(0)
    for images, labels in tqdm(data_loader):
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        
        logits, i, o  = model(images)
        if softmax_scores:
            score = torch.softmax(logits, dim=1)
        else:
            #i = torch.clamp(i, min=-max_ratio, max=max_ratio)
            #o = torch.clamp(o, min=-max_ratio, max=max_ratio)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)


        if get_acc:
            total += len(images)
            total_correct += torch.sum(logits.max(dim=1)[1].cpu() == labels)
        
        results1.extend(logits.data.cpu().numpy())
        results2.extend(i.data.cpu().numpy())
        results3.extend(o.data.cpu().numpy())
        labs.extend(labels.numpy())

        #z = model.z 
        zs = torch.cat([zs,  model.z.cpu().squeeze()], dim = 0)
  

    if get_acc:
        accuracy = float(total_correct) / total
        print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
        return accuracy,  (np.array(results1),np.array(results2),np.array(results3) ),  np.array(labs).squeeze(), zs.numpy()
    return (np.array(results1),np.array(results2),np.array(results3) ),  np.array(labs).squeeze(), zs.numpy()
    
#godin 0.94911 auc 0.98476 and tnr@tpr95 0.92090
if __name__ == '__main__':
    main()
