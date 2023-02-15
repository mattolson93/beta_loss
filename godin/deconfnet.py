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
class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x,w.T))
        return ret

class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = x.unsqueeze(2) #(batch, latent, 1)
        h = self.h.weight.T.unsqueeze(0) #(1, latent, num_classes)
        ret = -((x -h).pow(2)).mean(1)
        return ret
        
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)


class RatioDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioDeconf, self).__init__()

        self.h  = nn.Linear(in_features, num_classes)
        self.h1 = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.h1.weight.data, nonlinearity = "relu")
        self.h.bias.data  = torch.zeros(size = self.h.bias.size())
        self.h1.bias.data = torch.zeros(size = self.h1.bias.size())

    def forward(self, x):
        i = self.h(x)
        o = self.h1(x) 
        if self.training:
            i = torch.clamp(i ,min=-50, max=50)
            o = torch.clamp(o,min=-50, max=50)
            logits = torch.sigmoid(i) / (torch.sigmoid(o)+ 1e-7)
        else:
            logits = torch.sigmoid(torch.clamp(i ,min=-5000, max=5000)) / (torch.sigmoid(o+.000001) +.000001)
        

        return logits, i, o


'''class RatioDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioDeconf, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.h1.weight.data, nonlinearity = "relu")
        self.h.bias.data  = torch.zeros(size = self.h.bias.size())
        self.h1.bias.data = torch.zeros(size = self.h1.bias.size())

    def forward(self, x):

        i = self.inlier(x)
        o = self.outlier(x)
        logits = torch.sigmoid(i) / torch.sigmoid(o)

        return logits'''


class DeconfNet(nn.Module):
    def __init__(self, underlying_model, in_features, num_classes, h, baseline, cvae=None):
        super(DeconfNet, self).__init__()
        
        self.num_classes = num_classes

        self.underlying_model = underlying_model
        
        self.h = h
        
        self.baseline = baseline

        if baseline:
            self.g = lambda a : 1
        else:
            self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        
        self.softmax = nn.Softmax()
        self.cvae = cvae
        if cvae is not None:
            self.compress = nn.Linear(128*num_classes, in_features)

    def _forward(self, z):
        if type( self.h) == RatioDeconf:
            h = self.h.h(z)
            g = self.h.h1(z)
            logit = torch.sigmoid(h) / (torch.sigmoid(g+.000001) +.000001)
            return logit, h, g
        else:
            numerators = self.h(z)
            denominators = self.g(z)
            quotients = numerators / denominators

        # logits, numerators, and denominators
        return quotients, numerators, denominators

    
    def forward(self, x):
        output = self.underlying_model(x)
        if self.cvae is not None:
            c_zs = self.cvae(x, None, only_mu=True)
            c_zs = F.elu(self.compress(c_zs))
            output = torch.cat([c_zs,output], dim=1)

        self.z = output

        
        # Now, broadcast the denominators per image across the numerators by division.
        if type( self.h) == RatioDeconf:
            return self.h(output)
        else:
            numerators = self.h(output)
            denominators = self.g(output)
            quotients = numerators / denominators

        # logits, numerators, and denominators
        return quotients, numerators, denominators
