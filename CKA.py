import math
import numpy as np
import sys
import torch
from sklearn.decomposition import PCA

from energy import EnergyDistance

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit("need 2 cmd args")
    device = torch.device('cuda')
    #device = torch.device('cpu')

    xname = sys.argv[1]
    yname = sys.argv[2]
    num_items = int(sys.argv[3]) if len(sys.argv) == 4 else None
    X = torch.load(xname)[:num_items]
    Y = torch.load(yname)[:num_items]#[:10000]
    print(X.shape)
    #print(xname, yname)
    #print(np.linalg.norm(X-Y,ord=2, axis=1).mean())
    #pretrained vs dre = 28.299171
    #exit()
    #import pdb; pdb.set_trace()
    do_verbose = True


    d = EnergyDistance(X, Y, gpu=True)
    print("energy distance = ", d)

    cuda_cka = CudaCKA(device)

    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    #np.random.shuffle(Y)
   
    linear_result = cuda_cka.kernel_CKA(X, Y)
    print('Linear CKA, between X and Y: {}'.format(linear_result))
    


