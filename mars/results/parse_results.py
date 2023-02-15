import numpy as np
import os
import sys

infile = sys.argv[1]
data = np.loadtxt(infile, delimiter=',', dtype=np.str)
if len(data.shape) != 2:
	data =  np.expand_dims(data, axis=0)
data[:,2] = -1
data = data.astype(np.float)


#godin = ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
#splinear= ['max', 'ce', 'norm']
#{exp_id},{acc:.4f},{best_mode},{best_auc:.4f},{best_tnr:.4f} #0,1,2,3,4

if "godin" in infile:
	column_titles = ['latent_norm','logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
elif "linear" in infile:
	column_titles = ['ce','max','latent_norm','latent_norm2']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
else:
	exit(f"bad file of {infile}")

#scoring_func = ["best"]
#auc_columns = [3]
#tnr_columns = [4]


if "_softmax" in infile:
	infile = infile.replace("_softmax","softmax")

#print(f"{infile} is the experiment")
out = ",".join(os.path.basename(infile)[:-4].split("_"))
for a, t, title in zip (auc_columns, tnr_columns, column_titles):
	auc_avg = data.mean(0)[a]
	auc_std = data.std(0)[a]
	tnr_avg = data.mean(0)[t]
	tnr_std = data.std(0)[t]

	print(out +f",{title},{auc_avg:.5f},{auc_std:.5f},{tnr_avg:.5f},{tnr_std:.5f}")
	#print(f'{title} = {auc_avg:.3f}pm{auc_std:.3f} {tnr_avg:.3f}pm{tnr_std:.3f}')
