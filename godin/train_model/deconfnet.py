# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

#self.weights = torch.nn.Parameter(torch.randn(size = (num_classes, in_features)) * math.sqrt(2 / (in_features)))

class GodinLayerSoftplus(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GodinLayerSoftplus, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        self.softplus = nn.Softplus()

        self.init_weights()

    def get_modes(): return ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x, clampval=50):
        denominators = self.g(x)

        x = norm(x)
        w = norm(self.h.weight)
        numerators = (torch.matmul(x,w.T))

        quotients = numerators / denominators

        if self.training:
            return self.softplus(torch.clamp(quotients, min=-clampval, max=clampval)) 
        else:
            return quotients, numerators, denominators
            
class GodinLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GodinLayer, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        self.init_weights()


    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        denominators = self.g(x)

        x = norm(x)
        w = norm(self.h.weight)
        numerators = (torch.matmul(x,w.T))

        quotients = numerators / denominators

        if self.training: 
            return quotients
        return quotients, numerators , denominators

class NormalLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(NormalLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        if self.training: 
            return self.h(x)
        return self.h(x),0,0

class SoftplusLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SoftplusLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        self.softplus = nn.Softplus()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x, clampval=50):
        if self.training: 
            return self.softplus(torch.clamp(self.h(x), min=-clampval, max=clampval))
        return self.softplus(self.h(x))


class RatioLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioLayer, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.in_featuressqrt = (in_features **.5)
        self.init_weights()


    def init_weights(self):
        nn.init.kaiming_normal_(self.inlier.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.outlier.weight.data, nonlinearity = "relu")
        self.inlier.bias.data  = torch.zeros(size = self.inlier.bias.size())
        self.outlier.bias.data = torch.zeros(size = self.outlier.bias.size())

    def forward(self, x, max_ratio=50):
        eps =  .00000001
        i = self.inlier(x) #/ self.in_featuressqrt
        o = self.outlier(x) #/ self.in_featuressqrt

        if self.training:
            i = torch.clamp(i, min=-max_ratio, max=max_ratio)
            o = torch.clamp(o, min=-max_ratio, max=max_ratio)
       

        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ eps)
        if self.training: 
            return logits
        return logits, i, o


class DeconfNet(nn.Module):
    def __init__(self, underlying_model, h):
        super(DeconfNet, self).__init__()
        self.underlying_model = underlying_model
        
        self.fc = h
    
    def forward(self, x):
        self.z = self.underlying_model(x)
        return self.fc(self.z)

