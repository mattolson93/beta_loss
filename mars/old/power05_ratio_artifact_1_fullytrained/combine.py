import numpy as np


val_labs = np.load("val_labs.npy")
valopen_labs = np.load("valopen_labs.npy")
valopen_zs = np.load("valopen_zs.npy")
val_zs = np.load("val_zs.npy")

open_labs = np.load("open_labs.npy")
open_zs = np.load("open_zs.npy")
test_labs = np.load("test_labs.npy")
test_zs = np.load("test_zs.npy")


val_labs = np.array([0]*len(val_labs))
valopen_labs = np.array([1]*len(valopen_labs))

test_labs = np.array([0]*len(test_labs))
open_labs = np.array([1]*len(open_labs))

np.save("out_trainzs.npy",       np.concatenate([val_zs, valopen_zs]))
np.save("out_trainlabs.npy",     np.concatenate([val_labs, valopen_labs]))
np.save("out_testzs.npy",       np.concatenate([test_zs, open_zs]))
np.save("out_testlabs.npy",     np.concatenate([test_labs, open_labs]))
