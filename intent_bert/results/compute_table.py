import numpy as np

import sys

data = np.loadtxt(sys.argv[1], delimiter=',')
outstring = ''
for mean, std in zip(data.mean(0)[1:], data.std(0)[1:]):
    outstring+=f'{mean:.3f}pm{std:.3f} '
print(outstring[:-1])

outstring = ''
for mean, std in zip(data.mean(0)[1:], data.std(0)[1:]):
    outstring+=f'{mean:.3f} '
print(outstring[:-1])
    