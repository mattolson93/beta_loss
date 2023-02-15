import numpy as np
import os
import sys
from glob import glob
from collections import defaultdict


def parse_csv(out, data, a, t, title): 
	acc_avg = data.mean(0)[2]
	acc_std = data.std(0)[2]
	auc_avg = data.mean(0)[a]
	auc_std = data.std(0)[a]
	tnr_avg = data.mean(0)[t]
	tnr_std = data.std(0)[t]
	print(out +f",{title},{acc_avg:.5f},{acc_std:.5f},{auc_avg:.5f},{auc_std:.5f},{tnr_avg:.5f},{tnr_std:.5f}")






indir = sys.argv[1]
best_only = True

dirs = np.unique(np.array([d[:-2] for d in glob(indir+"/*")]))

for exp_set in dirs:
	#if "cifarall" not in exp_set and "cifar100" not in exp_set: continue
	datas = defaultdict(list)
	split_dirs = glob(exp_set + "*")
	for d in split_dirs: #e.g. 5 default godin exps on tinyimagenet
		result_files = glob(d + "/*final.csv")
		for f in result_files: #1 individual csv
			try:
				#if "fixed" in f: continue
				#import pdb; pdb.set_trace()
				f_data = np.loadtxt(f, delimiter=',', dtype=np.str)
				f_data[3] = -1
				#datas[f.split("/")[-1][-1]].append(f_data.astype(np.float))
				datas[f.split("/")[-1]].append(f_data.astype(np.float))
			except:
				pass
				#import pdb; pdb.set_trace()


	if best_only:
		scoring_func = ["best"]
		auc_columns = [4]
		tnr_columns = [5]
	else:
		if "godin" in exp_set:
			scoring_func = ["logit_norm","h_norm","g_norm","latent_norm","latent_norm2","h_max","logit_max","g_max"]
		else:
			scoring_func = ["logit_ce","logit_max","logit_norm","latent_norm","latent_norm2"]

		scoring_func = ["acc","best"] + scoring_func

		auc_columns = np.concatenate([np.array([2,4]) , (np.arange(len(scoring_func))*2) + 6])
		tnr_columns = np.concatenate([np.array([2,5]) , (np.arange(len(scoring_func))*2) + 7])


	for k in datas.keys():
		out =  ",".join(os.path.basename(exp_set).split("_")[:-1]) + "," + k[:-10]
		data = np.stack(np.array(datas[k]))


		for a, t, title in zip (auc_columns, tnr_columns, scoring_func):
			parse_csv(out, data, a, t, title)
			



exit()


data = np.loadtxt(infile, delimiter=',', dtype=np.str)
if len(data.shape) != 2:
	data =  np.expand_dims(data, axis=0)
data[:,2] = -1
data = data.astype(np.float)


#godin = ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
#splinear= ['max', 'ce', 'norm']
#{exp_id},{acc:.4f},{best_mode},{best_auc:.4f},{best_tnr:.4f} #0,1,2,3,4
if "ce_linear" in infile:
	column_titles = ["logit_ce"]
	auc_columns = [7]
	tnr_columns = [8]
	#column_titles = ["logit_ce","logit_max"]
	#auc_columns = [5,7]
	#tnr_columns = [6,8]
elif "linearspr" in infile:
	column_titles = ['ce','max','latent_norm']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
elif "linearsp" in infile:
	column_titles = ["max", "norm"]
	auc_columns = [5,9]
	tnr_columns = [6,10]
elif "godin" in infile:
	column_titles = ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
elif "dreratio" in infile:
	column_titles = ['latent_norm','logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
elif "ratio" in infile:
	column_titles = ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']
	auc_columns = (np.arange(len(column_titles))*2) +5
	tnr_columns = (np.arange(len(column_titles))*2) + 6
else:
	exit(f"bad file of {infile}")

#print(f"{infile} is the experiment")
out = ",".join(os.path.basename(infile)[:-4].split("_"))
for a, t, title in zip (auc_columns, tnr_columns, column_titles):
	auc_avg = data.mean(0)[a]
	auc_std = data.std(0)[a]
	tnr_avg = data.mean(0)[t]
	tnr_std = data.std(0)[t]

	print(out +f",{title},{auc_avg:.5f},{auc_std:.5f},{tnr_avg:.5f},{tnr_std:.5f}")
	#print(f'{title} = {auc_avg:.3f}pm{auc_std:.3f} {tnr_avg:.3f}pm{tnr_std:.3f}')
