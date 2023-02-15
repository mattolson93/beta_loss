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
from torch.nn.functional import log_softmax, nll_loss
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import argparse
import os
import shutil
import glob
#from resnet import ResNet34,ResNet18, ResNet152

from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from deconfnet import DeconfNet, NormalLinear, GodinLayer, RatioLayer, SoftplusLinear, GodinLayerSoftplus, ForcedPositive, RatioLayer2
from datasets import OpenDatasets

from grad_clamp import dclamp
from focal import focal_loss

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


eps = np.finfo(np.float32).eps 
one_m_eps = 1 - eps
logit_range = -np.log(eps)

def get_optimizers(params, h_params, choice, lr):
    if choice == 'sgd':
        optimizer = optim.SGD(params, lr = lr, momentum = 0.9, weight_decay = 0.0001)
        h_optimizer = optim.SGD(h_params, lr = lr, momentum = 0.9, weight_decay = 0) # No weight decay
    elif choice == 'adam':
        optimizer = optim.Adam(params, lr = lr)
        h_optimizer = optim.Adam(h_params, lr = lr, weight_decay = 0) # No weight decay
    elif choice == 'adamw':
        optimizer = optim.AdamW(params, lr = lr)
        h_optimizer = optim.AdamW(h_params, lr = lr, weight_decay = 0) # No weight decay
    elif choice == 'adadelta':
        optimizer = optim.Adadelta(params, lr = lr)
        h_optimizer = optim.Adadelta(h_params, lr = lr) # No weight decay
    elif choice == 'adagrad':
        optimizer = optim.Adagrad(params, lr = lr)
        h_optimizer = optim.Adagrad(h_params, lr = lr) # No weight decay
    elif choice == 'adamax':
        optimizer = optim.Adamax(params, lr = lr)
        h_optimizer = optim.Adamax(h_params, lr = lr) # No weight decay
    elif choice == 'asgd':
        optimizer = optim.ASGD(params, lr = lr)
        h_optimizer = optim.ASGD(h_params, lr = lr) # No weight decay
    elif choice == 'LBFGS':
        optimizer = optim.LBFGS(params, lr = lr)
        h_optimizer = optim.LBFGS(h_params, lr = lr) # No weight decay
    elif choice == 'rms':
        optimizer = optim.RMSprop(params, lr = lr)
        h_optimizer = optim.RMSprop(h_params, lr = lr) # No weight decay
    elif choice == 'rprop':
        optimizer = optim.Rprop(params, lr = lr)
        h_optimizer = optim.Rprop(h_params, lr = lr) # No weight decay
    elif choice == 'adabound':
        import adabound
        optimizer = adabound.AdaBound(params, lr = lr, final_lr=0.1)
        h_optimizer = adabound.AdaBound(h_params, lr = lr, final_lr=0.1) # No weight decay
    else:
        exit("invalid optimizer chocie")

    return optimizer, h_optimizer

def save_results(directory, outstrings, outdatasets, outfile="results.csv"):
    for outstring, d in zip(outstrings, outdatasets):
        with open(os.path.join(directory, d + "_" + outfile), "a") as writer:
            writer.write(outstring+"\n")

def main():
    args = get_args()
    
    device           = "cuda"

    layer_type       = args.layer_type
    loss_type        = args.loss_type
    
    data_dir         = args.data_dir
    data_name        = args.out_dataset
    batch_size       = args.batch_size
    
    weight_decay     = args.weight_decay
    lr               = args.lr
    epochs           = args.epochs
    seed             = args.seed

    info             = args.info
    do_train         = args.train

    alpha            = args.alpha
    beta             = args.beta
    clip             = args.clip
    factivation      = args.fact

    do_clamp      = args.clamp
    optim_type       = args.optim


    if not do_clamp:
        print("not clamping logits")
        global logit_range
        logit_range = np.inf
   


    #if layer_type == "linear" and loss_type != 'ce': exit("linear layer can only be used with CrossEntropyLoss")
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.opencat_exp == "cifar100":
        dataset_name= args.opencat_exp
        train_data, test_data, open_data, outdatasets = get_datasets(data_dir, data_name, batch_size, do_cifar100=True)
        num_classes = 100
    elif args.opencat_exp != "None":
        dataset_name = args.opencat_exp
        train_data, test_data, open_data, outdatasets, num_classes = get_opencat(data_dir, args.opencat_exp, batch_size, seed)
    else:
        #get outlier data
        dataset_name = 'cifarall'
        num_classes = 10
        train_data, test_data, open_data, outdatasets = get_datasets(data_dir, data_name, batch_size)


    train_data_unshuffled = DataLoader(train_data,      batch_size=batch_size, shuffle=False,  num_workers=4)
    train_data = DataLoader(train_data,      batch_size=batch_size, shuffle=True,  num_workers=4)
    # Create necessary directories
    exp_name = f'{loss_type}_{alpha}_{beta}_c{clip}_clamp{do_clamp}_opti{optim_type}_lr{lr}_{factivation}_{layer_type}_{dataset_name}_{seed}_{info}'

    model_dir        = os.path.join(args.model_dir,exp_name)

    print("running experiment of name: ",exp_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        for filename in glob.glob(os.path.join(".", '*.py')):
            shutil.copy(filename, model_dir)

    CustomLayers = {
        'linear': NormalLinear,
        'godin': GodinLayer,
        'ratio':RatioLayer,
        'ratio2':RatioLayer2,
        'linearsp': SoftplusLinear,
        'godinsp': GodinLayerSoftplus,
        'positive': ForcedPositive,
    }


    underlying_net = DenseNet3(depth = 100, num_classes = num_classes).to(device)
  
    last_layer = CustomLayers[layer_type](underlying_net.output_size, num_classes).to(device)


    optimizer, h_optimizer = get_optimizers(underlying_net.parameters(), last_layer.parameters(), optim_type, args.lr)

    
    #from madgrad_wd import madgrad_wd
    model = DeconfNet(underlying_net, last_layer).to(device)
  
    mylosses = losses(factivation, args.loss_model_dir)
    if "model" in loss_type:
        mylosses.set_lossmodel(alpha, beta)
        criterion = mylosses.get_lossmodel
    elif "beta" in loss_type:
        criterion = mylosses.getset_alpha_beta(alpha, beta, device)
    else:
        criterion = mylosses.get_loss_dict()[loss_type]
    epoch_loss = None
    outmodel_pth = os.path.join(model_dir,'model.pth')

    header_str = 'epoch,seed,test_only,ood_only,both_auc'
    
    model.load_state_dict(torch.load("./model.pth"))  
    if do_train:
        model.train()
        
        num_batches = len(train_data)
        epoch_bar = tqdm(total = num_batches * epochs, initial = 0)
        epoch_loss = -42.
        
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            loss_float = -42.
            if epoch >= args.eval_start  and epoch % 5 == 0:
                cur_strs = eval_model(model,test_data, [open_data[0]], layer_type )
                cur_strs = [f"{seed},{epochs},{c}" for c in cur_strs]

                save_results(model_dir, cur_strs, [outdatasets[0]])
                model.train()
                for c in cur_strs: print(f"{epoch} results {c}")
                print(f'epoch| epoch_loss | cur_loss |  max_logit |  min_logit | used | max_inlier_loss | max_outlier_loss | batch')

            logits_saved = []
            labs = []

            for batch_idx, (inputs, targets) in enumerate(train_data):
                labs.extend(targets.numpy())

                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()
                
                logits = model(inputs)
                #rand_alpha, rand_beta = (-torch.rand(2)).cuda()
                #mylosses.getset_alpha_beta(rand_alpha, rand_beta, device)
                loss = criterion(logits, targets)

                inlier_loss  = mylosses.inlier_loss
                outlier_loss = mylosses.outlier_loss

                if loss != loss: exit("nans")
                loss.backward()
                
                if clip > 0 : torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                optimizer.step()
                h_optimizer.step()
                loss_float = loss.item()
                total_loss += loss_float

                max_logit = torch.max(logits).item()
                min_logit = torch.min(logits).item()
                max_inlier_loss = torch.max(inlier_loss).item()
                max_outlier_loss = torch.max(outlier_loss).item()
                used_logits = mylosses.used_logits 
                #grads = model.fc.h.bias.grad.sum().item()

                #####epoch_bar.set_description(f'{epoch + 1}/{epochs}| {epoch_loss:0.2f} | {loss_float:.1f} | b{batch_idx + 1}/{num_batches}')
                #print(f'{epoch + 1}/{epochs}| {epoch_loss:0.2f} | {loss_float:.2f} | {max_logit:.1f} |  {min_logit:.1f} | {used_logits } | {max_inlier_loss:.2f} | {max_outlier_loss:.2f} | b{batch_idx + 1}/{num_batches}')
                epoch_bar.set_description(f'{epoch + 1}/{epochs}| {epoch_loss:0.2f} | {loss_float:.2f} | {max_logit:.1f} |  {min_logit:.1f} | {used_logits } | {max_inlier_loss:.2f} | {max_outlier_loss:.2f} | b{batch_idx + 1}/{num_batches}')
                
            #logits_saved = np.array(logits_saved).squeeze()
            #labs = np.array(labs).squeeze()
            #acc, _, _,  _ = testData_getinfo(model, device, test_data , 0, criterion, score_func, title = 'Testing ID', get_acc=True) 
            #print("acc:", acc)
            epoch_loss = total_loss
            #h_scheduler.step()
            #scheduler.step()
        
            if epoch == 149 or  epoch ==174:
                print("saving model to ", outmodel_pth)    
                torch.save(model.state_dict(), os.path.join(model_dir,f'model{epoch}.pth') ) # For exporting / sharing / inference only
                torch.save(optimizer.state_dict(), os.path.join(model_dir,f'optimizer{epoch}.pth') ) # For exporting / sharing / inference only
                #  torch.save(h_optimizer.state_dict(), os.path.join(model_dir,f'h_optimizer{epoch}.pth') ) # For exporting / sharing / inference only
        
        print("training finished. Saving model")
        torch.save(model.state_dict(), os.path.join(model_dir,'model.pth') ) # For exporting / sharing / inference only
    else:
        model.load_state_dict(torch.load(outmodel_pth))  
        #model.load_state_dict(torch.load("adam/model_-0.5_-0.4_lr0.001_logit_linear_cifarall_0_noclamp/model149.pth"))  
        #model.load_state_dict(torch.load("outmodels/model-a5b4_-0.5_-0.4_c5.0_maxloss1000_dre_ratio2_cifarall_0_/model.pth"))  
   
    if True:
        torch.save(get_embeddings(model, train_data_unshuffled).numpy(), os.path.join(model_dir,'trainzs.pth') ) 
        torch.save(get_embeddings(model, test_data).numpy(), os.path.join(model_dir,'testzs.pth') ) 
        torch.save(get_embeddings(model, open_data[0]).numpy(), os.path.join(model_dir,'openzs.pth') ) 
             
    beststr = eval_model(model,test_data, open_data, layer_type )
    beststr = [f"{seed},{epochs},{b}" for b in beststr]
    
    for b in beststr: print("results: ", b)
    save_results(model_dir, [beststr[0]], [outdatasets[0]])
    save_results(model_dir, beststr, outdatasets, "final.csv")

    


@torch.no_grad()
def get_embeddings(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    zs = torch.empty(0)

    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch
        model(batch[0].to(device))
        z = model.z1
        zs = torch.cat([zs, z.cpu()], dim = 0)

    return zs



def eval_model(model,test_data, open_data, layer_type ):
    model.eval()

    if "godin" in layer_type:
        scoring_func = ["logit_norm","h_norm","g_norm","latent_norm","latent_norm2","h_max","logit_max","g_max"]
    elif "linear" in layer_type or "positive" in layer_type:
        scoring_func = ["logit_ce","logit_max","logit_norm","latent_norm","latent_norm2"]
    elif "ratio" in layer_type:
        scoring_func = ["logit_norm","h_norm","g_norm","latent_norm","latent_norm2","h_max","logit_max","g_max"]
        #scoring_func = ["latent_h"]
    else:
        exit("bad layer type")

    
    outstr = save_openset_all(model, test_data, open_data, scoring_func)

    return outstr



def get_score(preds, mode):
    if mode  in ['ce','h_ce','g_ce','logit_ce']:# 'confidence_threshold':
        return torch.max(torch.softmax(preds, dim=1), dim=1)[0].data.cpu().numpy()
    elif mode == 'augmented_classifier':
        return preds[:, -1].data.cpu().numpy()
        #return torch.softmax(preds, dim=1)[:, -1].data.cpu().numpy()
    elif mode == 'entropy':
        return ((preds * torch.log(preds)).sum(1)).data.cpu().numpy()
    elif mode  in ['max','h_max','g_max','logit_max']:
        return torch.max(preds, dim=1)[0].data.cpu().numpy()
    elif mode  in ['norm','h_norm','g_norm','logit_norm','latent_norm']:
        #return torch.mean(preds, dim=1).data.cpu().numpy()
        return torch.norm(preds,p=1, dim=1).data.cpu().numpy()
    elif mode  in ['latent_norm2']:
        #return torch.mean(preds, dim=1).data.cpu().numpy()
        return torch.norm(preds,p=2, dim=1).data.cpu().numpy()
    else:
        print("bad mode of ", mode)
        exit()

@torch.no_grad()
def get_model_preds(model, dataset, modes, get_acc=False):
    device = next(model.parameters()).device
    ret = {}
    for m in modes: ret[m] = []
    total_correct = 0
    total = 0

    for batch in dataset:
        if len(batch) == 2:
            images, labels = batch
            logits = model(images.to(device))
        else:
            input_ids, input_mask, segment_ids, labels = batch
            logits = model(input_ids.to(device), segment_ids.to(device), input_mask.to(device))
        
        if get_acc:
            preds = logits[0] if '_' in modes[0] else logits
            total_correct += torch.sum(preds.max(dim=1)[1].cpu() == labels)
            total+=preds.size(0)



        for m in modes: 
            if 'latent' in m:
                score = get_score(model.z, m)
            elif 'logit_' in m:
                score= get_score(logits[0], m)
            elif 'h_' in m:
                score= get_score(logits[1], m)
            elif 'g_' in m:
                score= get_score(logits[2], m)
            else:
                score= get_score(logits, m)
            ret[m].extend(score)
    
    if get_acc: return ret, (float(total_correct)/total  )  
    return ret



@torch.no_grad()
def save_openset_all(model, testing_dataset, openset_dataset, modes):
    known_scores, acc = get_model_preds(model, testing_dataset, modes, get_acc=True)
    unknown_scores_list = [get_model_preds(model, openset, modes) for openset in openset_dataset]

    outstrs = []
    for unknown_scores in unknown_scores_list:
        best_auc = 0
        outstr = ""
        for m in modes:
            k, u = known_scores[m], unknown_scores[m]
            if calc_auroc(k, u) < .5:
                u, k = known_scores[m], unknown_scores[m]

            auc = calc_auroc(k, u) 
            tnr = calc_tnr(k, u) 
            outstr += f',{auc:.4f},{tnr:.4f}'
            if auc > best_auc:
                best_auc = auc
                best_tnr = tnr
                best_mode = m

            print(f'{auc:.4f} AUC, {tnr:.4f} tnr.  Mode {m}') 
            print(f'avg   known:  {np.mean(known_scores[m]):.4f}~ {np.std(known_scores[m]):.4f}')
            print(f'avg unknown:  {np.mean(unknown_scores[m]):.4f}~ {np.std(unknown_scores[m]):.4f}')
        
        best_str = f'{acc:.4f},{best_mode},{best_auc:.4f},{best_tnr:.4f}'
        outstr = best_str + outstr


        print('id,acc,best_mode,best_auc,best_tnr,',modes)
        print("best,e,   acc, mode, auc, tnr")
        print("best",best_str)
        outstrs.append(outstr)
    return outstrs

  





class Net(torch.nn.Module):
    def __init__(self, is_prob):
        super(Net, self).__init__()
        dim = 256
        self.is_prob = is_prob
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, 1)
        
    def forward(self, x):
        inshape = x.shape
        x = x.flatten().unsqueeze(1)
        if self.is_prob:
            x = torch.log(x/(1-x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc4(x)
        return torch.exp(x).squeeze().reshape(inshape)


class losses:
    def __init__(self, output_type, mlp_model_dir):
        self.softplus = nn.Softplus() #if do_softplus else lambda x: x
        #self.clip1 =  -torch.log(torch.tensor(.90)).cuda()
        #self.clip2 =  -torch.log(torch.tensor(.95)).cuda()
        #self.clip3 =  -torch.log(torch.tensor(.98)).cuda()
        #self.clip4 =  -torch.log(torch.tensor(.99)).cuda()
        self.max_clip = torch.tensor(100).float().cuda()

        self.init()
        self.do_sigmoid=True
        self.used_logits = "   "
        self.fact = output_type
        self.inlier_loss = torch.tensor(0)
        self.outlier_loss = torch.tensor(0)
        self.mlp_model_dir = mlp_model_dir
        

        self.layer_dict = {
            'sigmoid': self.last_sigmoid,
            'softmax': self.last_softmax,
            'identity': self.last_identity,
            'dre': self.last_dre,
            'boost': self.last_boost,
            'logit': self.last_logit,
            'none': self.error
        }
        self.last_layer_type = output_type
        self.last_layer = self.layer_dict[output_type]
        self.bce_crit = nn.BCEWithLogitsLoss()


    def set_lossmodel(self,a,b):
        is_prob = not (self.fact == "logit" or self.fact == "noclamp")
        print("converting probs to logits = ", is_prob)
        self.infunc_model = Net(is_prob).cuda()
        self.infunc_model.load_state_dict(torch.load(os.path.join(self.mlp_model_dir, f"a{a}b{b}_L1.pt")))
        self.outfunc_model = Net(is_prob).cuda()
        self.outfunc_model.load_state_dict(torch.load(os.path.join(self.mlp_model_dir, f"a{a}b{b}_L0.pt")))
    
    def error(self, x): exit("bad flayer")

    def last_logit(self, logits):
        return  dclamp(logits,min=-logit_range, max=logit_range)
    def last_dre(self, logits, labels=None):
        logits =  dclamp(logits,min=-logit_range, max=logit_range)

        return logits / (logits+1)

    def last_sigmoid(self, logits):
        logits =  dclamp(logits,min=-logit_range, max=logit_range)
        return torch.sigmoid(logits)

    def last_softmax(self, logits):
        #logits =  dclamp(logits,min=-logit_range, max=logit_range)
        return torch.softmax(logits,dim=1)

    def last_identity(self, logits):
        logits =  dclamp(logits,min=-logit_range, max=logit_range)
        return logits

    def last_boost(self, logits):
        f =  dclamp(logits,min=-logit_range, max=logit_range)
        half_f = f * .5

        return .5* ((half_f / torch.sqrt((half_f**2) + 1)) + 1)
        
 
    def get_lossmodel(self, logits, labels):
        probs = self.last_layer(logits)
        
        y = torch.eye(probs.size(1))
        labels = y[labels].to(probs.device)
        
        if self.last_layer_type == "noclamp" or self.last_layer_type == "softmax":
            use_logit_in  = labels
            use_logit_out = (1-labels) 
        else:
            use_logit_in  = labels     * (logits <  logit_range).int() if self.fact != 'dre' else labels     * (logits <  499.999999).int()
            use_logit_out = (1-labels) * (logits > -logit_range).int() if self.fact != 'dre' else (1-labels) * (logits >= eps).int()
        



        inlier_loss  = self.infunc_model(probs) * use_logit_in
        outlier_loss = self.outfunc_model(probs) * use_logit_out

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean() 
        #loss = (inlier_loss + outlier_loss).mean()
        return loss




    def log_loss(self, logits, labels, max_ratio=15.942385):
        preds = self.last_layer(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        
        #print(preds[0])
        infunc  = -torch.log(preds )
        outfunc = -torch.log(1.0-preds)

        
        self.inlier_loss  = (labels     * infunc )
        self.outlier_loss = ((1-labels) * outfunc)
        loss = (self.inlier_loss + self.outlier_loss).mean()#/preds.size(1)
        return loss
    
    def squared_loss(self, logits, labels, max_ratio=15.942385):
        preds = self.last_layer(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        
        infunc  = (1-preds)**2
        outfunc = preds**2

        inlier_loss  = (labels     * infunc ).mean(1)
        outlier_loss = ((1-labels) * outfunc).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss
    
    def boosting_loss(self, logits, labels, max_ratio=15.942385):
        preds = self.last_layer(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        
        infunc  = torch.sqrt((1-preds)/(preds + eps))
        outfunc = torch.sqrt(preds /(1-preds + eps))

        use_logit_in  = labels     * (logits <  logit_range).int()
        use_logit_out = (1-labels) * (logits > -logit_range).int()

        inlier_loss  = dclamp(infunc * use_logit_in, max = self.max_clip, min=None)
        outlier_loss = dclamp(outfunc * use_logit_out, max = self.max_clip, min=None)
 

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean()

        return loss
    
    def halves_loss(self, logits, labels, max_ratio=15.942385):
        preds = self.last_layer(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        
        infunc  = torch.asin(torch.sqrt(1-preds) - eps*10)-torch.sqrt(preds*(1-preds + eps))
        outfunc = torch.asin(torch.sqrt(preds )- eps*10)-torch.sqrt(preds*(1-preds + eps))

        inlier_loss  = (labels     * infunc ).mean(1)
        outlier_loss = ((1-labels) * outfunc).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss
    
    def twos_loss(self, logits, labels, max_ratio=15.942385):
        preds = self.last_layer(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        
        infunc  = ((1./3.0)*(1-preds + eps)**3) - .25*(1-preds + eps)**4
        outfunc = ((1./3.0)*(  preds + eps)**3) - .25*(  preds + eps)**4

        inlier_loss  = (labels     * infunc ).mean(1)
        outlier_loss = ((1-labels) * outfunc).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss


    def kliep_loss(self, logits, labels, max_ratio=15.942385):
        preds = dclamp(logits,min=-1*max_ratio, max=max_ratio)
        

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

    def ulsif_loss(self, logits, labels, max_ratio=15.942385):
        preds = dclamp(logits,min=-1*max_ratio, max=max_ratio)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (-2*(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds**2)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss


    def power_loss(self, logits, labels, alpha=.1, max_ratio=15.942385):
        preds = dclamp(logits,min=-1*max_ratio, max=max_ratio)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        use_logit_in  = labels     * (preds <  max_ratio).int()
        use_logit_out = (1-labels) * (preds > 2e-7).int()



        inlier_loss  = dclamp(use_logit_in * (1 - preds.pow(alpha))/(alpha), self.max_clip).sum(1)
        outlier_loss = dclamp(use_logit_out * (preds.pow(1+alpha)-1)/(1+alpha), self.max_clip).mean(1)

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def power_loss_05(self, logits, labels): return self.power_loss(logits, labels, alpha=.05, max_ratio=15.942385)
    def power_loss_10(self, logits, labels): return self.power_loss(logits, labels, alpha=.1 , max_ratio=15.942385)
    def power_loss_50(self, logits, labels): return self.power_loss(logits, labels, alpha=.5 , max_ratio=15.942385)
    def power_loss_90(self, logits, labels): return self.power_loss(logits, labels, alpha=.90, max_ratio=15.942385)


    def getset_alpha_beta(self, alpha, beta, device):
        self.alpha = torch.tensor(alpha, requires_grad=False).to(device)
        self.beta  = torch.tensor(beta, requires_grad=False).to(device)


        if alpha == -42 and beta == -42:
            return self.log_loss

        if alpha == -50 and beta == -50:
            return self.boosting_loss



        return self.trapz_loss


    def L1_prime(self, q,a,b):
        return (torch.pow(dclamp(q,min=eps),(a-1))) * torch.pow(dclamp(1-q,min=eps),b)
    
    def L0_prime(self, q,a,b):
        return (torch.pow(dclamp(q,min=eps),a))     * torch.pow(dclamp(1-q,min=eps),(b-1))
      
        
    
    def trapz_loss(self, logits, labels, max_ratio=15.942385):
        probs = self.last_layer(logits)
        
        y = torch.eye(probs.size(1))
        labels = y[labels].to(probs.device)
        
        num_tzoids = 10000

        a = self.alpha
        b = self.beta
        
        
        preds = probs.unsqueeze(0).repeat(num_tzoids,1,1)
        
        
        lower_preds = preds * torch.linspace(eps,1.0,num_tzoids).unsqueeze(1).unsqueeze(1).cuda()
        upper_preds = 1- ((1-preds) * torch.linspace(1.0,eps,num_tzoids).unsqueeze(1).unsqueeze(1).cuda())

        
        infunc  = torch.trapz(self.L1_prime(upper_preds,a,b), upper_preds, dim=0)
        outfunc = torch.trapz(self.L0_prime(lower_preds,a,b), lower_preds, dim=0)


        use_logit_in  = labels     * (logits <  logit_range).int() if self.fact != 'softmax' else labels
        use_logit_out = (1-labels) * (logits > -logit_range).int() if self.fact != 'softmax' else (1-labels)



        inlier_loss  = dclamp(infunc * use_logit_in, self.max_clip)
        outlier_loss = dclamp(outfunc * use_logit_out, self.max_clip)

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean() #if logits.shape[1] <= 10 else (inlier_loss + outlier_loss/10).mean() #/preds.size(1) 

        return loss



    def custom_ce(self, logits, labels):
        logp = log_softmax(logits,1) 


        #y = torch.eye(logp.size(1))
        #batch_loss1 = torch.masked_select(-logp, (y[labels].to(logp.device) ==1.0))

        batch_loss2 = nll_loss(logp, labels, reduction = 'none')
        reduces = batch_loss2 < self.clip4
        batch_loss2[reduces] *= .01
        loss = batch_loss2.mean()

        #loss = batch_loss.mean()
        return loss

    

    def step_weight_section_g(self, blogits, labels):
        blogits = dclamp(blogits,min=-50, max=50)
        aa = self.a
        bb = self.b
        h  = self.h 
        blogits = torch.sigmoid(blogits)

        loss = torch.tensor(0).float().to(blogits.device)

        for logits, lab in zip(blogits, labels):

            for i, logit in enumerate(logits):
                cur_loss = self.l_neg1_sectiong(logit, aa, bb, h)#aa[i], bb[i], h[i])
                if i == lab:
                    cur_loss = cur_loss - logit

                loss+= cur_loss

        return loss

    def init(self):
        a = self.a = .05
        b = self.b = .95
        h = self.h = .0100

        self.const1 = (a**2)*(1-h)/(2*(h**2))
        self.const2 = (h-1) * (((((a-b)**2)*h) - (2*(a**2)) + (2*a*b)) / (2*h))

        self.interval1 = a/h
        self.interval2 = (b + ((1-h)/h)*a )

        self.term1 = ((h-1)/h)*a
        self.term2 = (h-1)*(a-b)

    def l_neg1_sectiong(self, v, a, b, h):

        if v <= self.interval1:
            ret = (h/2)* (v**2)
        elif v <= self.interval2:
            ret = ((v**2)/2) + (self.term1*v) + self.const1
        else:
            ret = ((h/2)* (v**2)) + (self.term2*v) + self.const2

        return ret

    def get_focal_loss(self, input, target): return focal_loss(input, target, alpha=0.25, gamma=2.0, reduction='mean')
    def bce(self, input, labels): 
        y = torch.eye(input.size(1))
        labels = y[labels].to(input.device)
        return self.bce_crit(input,labels)


    def get_loss_dict(self):
        return {
            'ce':nn.CrossEntropyLoss(),
            'bce':self.bce,
            'custom':self.custom_ce,
            'step': self.step_weight_section_g,
            'kliep':   self.kliep_loss, 
            'ulsif':   self.ulsif_loss, 
            'power05': self.power_loss_05, 
            'power10': self.power_loss_10, 
            'power50': self.power_loss_50, 
            'power90': self.power_loss_90, 
            'log': self.log_loss,
            'square': self.squared_loss,
            'boost': self.boosting_loss,
            'half': self.halves_loss,
            'twos': self.twos_loss,
            'focal': self.get_focal_loss,
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
    parser.add_argument('--layer-type', default = 'ratio', type = str,
                        help = 'similarity function for decomposed confidence numerator (cosine | inner | euclid | baseline)')
    parser.add_argument('--loss-type', default = 'model', type = str,
                        help = 'ce|kliep|power05|power10|power50|power90|ulsif')
    parser.add_argument('--opencat-exp', default = 'None', type = str,
                        help = 'mnist|svhn|cifar10|cifar10+|cifar50+|tinyimagenet')

    # Data loading arguments
    parser.add_argument('--data-dir', default='./data', type = str)
    parser.add_argument('--loss-model-dir',  default='mlp_models', type = str)
    parser.add_argument('--info', default='', type = str)
    parser.add_argument('--out-dataset', default = 'Imagenet', type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--batch-size', default = 64, type = int,
                        help = 'batch size')
    parser.add_argument('--do-softplus', default = 1, type = int,
                        help = 'batch size')

    parser.add_argument('--lr', default = 0.1, type = float)

    parser.add_argument('--alpha', default = -0.5, type = float)
    parser.add_argument('--beta', default = -0.4, type = float)
    parser.add_argument('--clip', default = 5, type = float)
    parser.add_argument('--eval_start', default = 75, type = int)
    parser.add_argument('--fact', default = "dre", type = str)
    parser.add_argument('--optim', default = "sgd", type = str)


    # Training arguments
    parser.add_argument('--no-train', action='store_false', dest='train')
    parser.add_argument('--no-clamp', action='store_false', dest='clamp')
    parser.add_argument('--weight-decay', default = 0.0001, type = float,
                        help = 'weight decay during training')
    parser.add_argument('--epochs', default = 175, type = int,
                        help = 'number of epochs during training')
    parser.add_argument('--seed', default = 0, type = int,
                        help = 'number of epochs during training')

    # Testing arguments
    parser.add_argument('--no-test', action='store_false', dest='test')
    parser.add_argument('--magnitudes', nargs = '+', default = [0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                        help = 'perturbation magnitudes')
    
    
    parser.set_defaults(argument=True)
    return parser.parse_args()

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


def get_opencat(data_dir, oc_name, batch_size, seed=1):
    dataObject = OpenDatasets(batch_size, batch_size, num_workers=4, add_noisy_instances=False)

    train_loader, test_loader, openset_loader, num_classes   = dataObject.get_dataset_byname(oc_name, seed )  
    
    if oc_name == "cifar10+":
        _, _, openset2,_ = dataObject.get_dataset_byname("cifar50+", seed )  
        openset_loader = [openset_loader, openset2]
        outdatasets = [oc_name, "cifar50+"]
    else:
        openset_loader = [openset_loader]
        outdatasets = [oc_name]


    return train_loader, test_loader, openset_loader, outdatasets, num_classes


def get_datasets(data_dir, data_name, batch_size, do_cifar100=False):

    if do_cifar100:
        train_set_in = torchvision.datasets.CIFAR100(root=f'{data_dir}/cifar100', train=True, download=True, transform=train_transform)
        test_set_in  = torchvision.datasets.CIFAR100(root=f'{data_dir}/cifar100', train=False, download=True, transform=test_transform)
    else:       
        train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transform)
        test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transform)
    
    outdatasets = ["Imagenet", "Imagenet_resize", "iSUN", "LSUN", "LSUN_resize", "LSUN_resize_fixed", "Imagenet_resize_fixed"]
    
    outlier_loader = [DataLoader(torchvision.datasets.ImageFolder(f'{data_dir}/{d}', transform=test_transform), batch_size=batch_size, shuffle=True, num_workers=4) for d in outdatasets]

    
    #add svhn
    outlier_svhn    = torchvision.datasets.SVHN(data_dir+'/svhn', split='test',download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))  
    outlier_loader.append(DataLoader(outlier_svhn,       batch_size=batch_size, shuffle=False, num_workers=4))
    outdatasets.append("svhn" )
   
    #train_set_in = Subset(train_set_in, list(range(len(train_set_in)))[:10000])


    train_loader_in      =  train_set_in
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_in, test_loader_in, outlier_loader, outdatasets


   
if __name__ == '__main__':
    main()
