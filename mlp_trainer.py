#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
import torch.nn as nn
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

num_tzoids  =  250000
eps = np.finfo(np.float32).eps

logit_range = int(-np.log(eps)) 
print("eps max logit for float32 is: ", -np.log(eps))
aa = float(sys.argv[1])
bb = float(sys.argv[2])
alpha = torch.tensor(aa)
beta  = torch.tensor(bb)


def safesigmoid(logits, rangeval=logit_range): return torch.sigmoid(torch.clamp(logits, min=-rangeval, max=rangeval))

def L1_prime(q,a=alpha,b=beta):
        return (torch.pow(torch.clamp(q,min=eps),(a-1))) * torch.pow(torch.clamp(1-q,min=eps),b)

def L0_prime(q,a=alpha,b=beta):
    return (torch.pow(torch.clamp(q,min=eps),a))     * torch.pow(torch.clamp(1-q,min=eps),(b-1))
        
def partial_trapzloss1(logit):
    probs = torch.sigmoid(logit)
    preds = probs.unsqueeze(0).repeat(num_tzoids,1,1)

    d = 0.005
    up1_end = 1.0 - d

    up3_start = d

    up1 = 1- ((1-preds) * torch.linspace(1.0,up1_end,num_tzoids).unsqueeze(1).unsqueeze(1))
    up2 = (1- ((1-preds) * torch.linspace(up1_end,up3_start,num_tzoids).unsqueeze(1).unsqueeze(1)))[1:-1]
    up3 = 1- ((1-preds) * torch.linspace(up3_start,0.0,num_tzoids).unsqueeze(1).unsqueeze(1))



    upper_preds = torch.cat([up1,up2,up3],dim=0)
    return torch.trapz(L1_prime(upper_preds,alpha,beta), upper_preds, dim=0)

def partial_trapzloss0(logit):
    probs = torch.sigmoid(logit)
    preds = probs.unsqueeze(0).repeat(num_tzoids,1,1)

    d = 0.005
    up1_end = 1.0 - d

    up3_start = d

    up1 = preds * torch.linspace(0.0,up3_start,num_tzoids).unsqueeze(1).unsqueeze(1)
    up2 = (preds * torch.linspace(up3_start,up1_end,num_tzoids).unsqueeze(1).unsqueeze(1))[1:-1]
    up3 = preds * torch.linspace(up1_end,1.0,num_tzoids).unsqueeze(1).unsqueeze(1)


    upper_preds = torch.cat([up1,up2,up3],dim=0)
    return torch.trapz(L0_prime(upper_preds,alpha,beta), upper_preds, dim=0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 256
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))#, negative_slope=0.2,)
        x = F.relu(self.fc2(x))#, negative_slope=0.2,)
        return self.fc4(x) #+ .1*(self.fc5(x))  #+ .01*(self.fc6(x))



for loss_type in [0,1]:
    outmodel_file =  f"mlp_models/a{aa}b{bb}_L{loss_type}.pt"
    print("cur outmodel file: " ,  outmodel_file)

    xs = []
    ys = []
    loss_func = partial_trapzloss0 if loss_type == 0 else partial_trapzloss1

    for i in np.arange(-logit_range, logit_range, .01):
        logit = torch.tensor(float(i), dtype=torch.double)

        estloss = loss_func(logit)
                
        xs.append(logit.item())
        ys.append(estloss.item())

    xs = np.array(xs)
    ys = np.array(ys)

    model = Net().cuda()
    lr = .002
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #my_dataset = TensorDataset(torch.Tensor(xs).double().unsqueeze(1),torch.Tensor(ys).double().unsqueeze(1))
    my_dataset = TensorDataset(torch.Tensor(xs).unsqueeze(1).cuda(),torch.Tensor(ys).unsqueeze(1).cuda())
    my_dataloader = DataLoader(my_dataset, batch_size=5000)#, num_workers = 4, shuffle=True, pin_memory = True, drop_last=True) 
    criterion = nn.MSELoss()

    target_error = 1/(2*max(ys[0],ys[-1]))
    print("target_error: ",target_error)
    for i in range(50000):
        for x,y in my_dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - torch.log(y))**2) 
        
            loss.backward()
            optimizer.step()
            
        out1 = torch.exp(pred[0]).item()
        out2 = torch.exp(pred[-1]).item()
        cur_error = max(abs(out1 - ys[0]), abs(out2 - ys[-1]))
        if i % 50 == 0: 
            print(loss.item(), out1, ys[0], out2, ys[-1])
            '''if (prev_error - cur_error) < 0: 
                worse_count +=1
            else:
                worse_count = 0

            if worse_count == 5: break
            prev_error = cur_error'''
        if cur_error <= target_error: break


    torch.save(model.state_dict(), outmodel_file)

'''

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 1024
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))#, negative_slope=0.2,)
        x = F.relu(self.fc2(x))#, negative_slope=0.2,)
        x = F.relu(self.fc3(x))#, negative_slope=0.2,)
        return self.fc4(x) #+ .1*(self.fc5(x))  #+ .01*(self.fc6(x))

from tqdm import tqdm

for loss_type in [1,1]:
    outmodel_file =  f"mlp_models/big_a{aa}b{bb}_L{loss_type}.pt"
    print("cur outmodel file: " ,  outmodel_file)

    xs = []
    ys = []
    loss_func = partial_trapzloss0 if loss_type == 0 else partial_trapzloss1

    for  i in tqdm(np.arange(-logit_range, logit_range, .001)):
        logit = torch.tensor(float(i), dtype=torch.double)

        estloss = loss_func(logit)
                
        xs.append(logit.item())
        ys.append(estloss.item())



    xs = np.array(xs)
    ys = np.array(ys)

    np.save(f"xs{loss_type}.npy", xs)
    np.save(f"ys{loss_type}.npy", ys)

    xs = np.load(f"xs{loss_type}.npy")
    ys = np.load(f"ys{loss_type}.npy")

    model = Net().cuda()
    lr = .002
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #my_dataset = TensorDataset(torch.Tensor(xs).double().unsqueeze(1),torch.Tensor(ys).double().unsqueeze(1))
    my_dataset = TensorDataset(torch.Tensor(xs).unsqueeze(1).cuda(),torch.Tensor(ys).unsqueeze(1).cuda())
    my_dataloader = DataLoader(my_dataset, batch_size=50000)#, num_workers = 4, shuffle=True, pin_memory = True, drop_last=True) 
    criterion = nn.MSELoss()

    target_error = 1/(2*max(ys[0],ys[-1]))
    print("target_error: ",target_error)
    for i in tqdm(range(100000)):
        for x,y in my_dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - torch.log(y))**2) 
        
            loss.backward()
            optimizer.step()
            
        out1 = torch.exp(pred[0]).item()
        out2 = torch.exp(pred[-1]).item()
        cur_error = max(abs(out1 - ys[0]), abs(out2 - ys[-1]))
        if i % 50 == 0: 
            print(loss.item(), out1, ys[0], out2, ys[-1])
            if (prev_error - cur_error) < 0: 
                worse_count +=1
            else:
                worse_count = 0

            if worse_count == 5: break
            prev_error = cur_error
        #if cur_error <= target_error: break


    torch.save(model.state_dict(), outmodel_file)
    exit()


    '''