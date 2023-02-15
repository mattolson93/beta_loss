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
if "resnet18_ce" in infile:
	column_titles = ["best"]
	auc_columns = [5]
	tnr_columns = [6]
	#column_titles = ["logit_ce","logit_max"]
	#auc_columns = [5,7]
	#tnr_columns = [6,8]

else:
	column_titles = ['best','baseline_marginalized','baseline_n','baseline_m','baseline_norm','baseline_max', 'latent_norm']
	auc_columns = (np.arange(len(column_titles))*2) +3
	tnr_columns = (np.arange(len(column_titles))*2) + 4

#print(f"{infile} is the experiment")
cur_file = os.path.basename(infile)[:-4]
if "_resize" in cur_file:
	cur_file = cur_file.replace("_resize","resize")

if "_fixed" in cur_file:
	cur_file = cur_file.replace("_fixed","fixed")

out = ",".join(cur_file.split("_"))
for a, t, title in zip (auc_columns, tnr_columns, column_titles):
	acc_avg = data.mean(0)[1]
	acc_std = data.std(0)[1]
	auc_avg = data.mean(0)[a]
	auc_std = data.std(0)[a]
	tnr_avg = data.mean(0)[t]
	tnr_std = data.std(0)[t]

	print(out +f",{title},{acc_avg:.5f},{acc_std:.5f},{auc_avg:.5f},{auc_std:.5f},{tnr_avg:.5f},{tnr_std:.5f}")
	#print(f'{title} = {auc_avg:.3f}pm{auc_std:.3f} {tnr_avg:.3f}pm{tnr_std:.3f}')
