from model import BertForConstrainClustering
from utils import *
import argparse
import random
import torch
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import trange
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam#, warmup_linear
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from datetime import datetime
import warnings
warnings.warn = warn

results_all = {}
seed = 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


task_name = dataset = sys.argv[1]
fraction = float(sys.argv[2])
loss_type = sys.argv[3] #kliep, ce, original
if loss_type != "original" and "_" in loss_type:
    lam = float(loss_type.split("_")[-1])
else:
    lam = 1
seed = int(sys.argv[4])
full_task_name = "_".join(sys.argv[1:])

print("Starting task of: " + full_task_name)
data_dir = 'data/' + task_name
output_dir = 'outmod/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
bert_model = "bert-base-uncased"#"/data/disk1/sharing/pretrained_embedding/bert/uncased_L-12_H-768_A-12"
num_train_epochs = 30

max_seq_task = {
    "snips": 35,
    'dbpedia': 54,
    "stackoverflow": 20,
    "20ng": 128,
}
max_seq_length = max_seq_task[task_name]
train_batch_size = 512
eval_batch_size= 2048
learning_rate = 2e-5
warmup_proportion = 0.1

processors = {
    "snips": SnipsProcessor,
    'dbpedia': Dbpedia_Processor,
    "stackoverflow": Stackoverflow_Processor,
    "20ng": NewsGroup20Processor,
}

num_labels_task = {
    "snips": 7,
    'dbpedia': 14,
    "stackoverflow": 20,
    "20ng": 20,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))
logger.disabled = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if os.path.exists(output_dir) and os.listdir(output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
num_labels = num_labels_task[task_name]
label_list = processor.get_labels()
num_known_labels = round(num_labels*fraction)
print(f"total labels: {num_labels}, known_labels: {num_known_labels}")
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

# Freezing all transformer (except the last layer)
model = BertForConstrainClustering.from_pretrained(bert_model, num_labels = num_known_labels)
for name, param in model.bert.named_parameters():  
    param.requires_grad = False
    if "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train_examples = processor.get_train_examples(data_dir)
#num_train_optimization_steps = int((len(train_examples) / train_batch_size) * num_train_epochs * (1+labeled_ratio)) + 1
num_train_optimization_steps = int((len(train_examples) / train_batch_size) * num_train_epochs) + 1
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

## Settings of unknown classes discovery
# 1. Select 25% classes as unknown(-1)
# 2. Set 90% of examples as unknown(-1)

classes = list(range(0,num_labels))
random.shuffle(classes)
INLIER = classes[:num_known_labels]
OUTLIER = classes[num_known_labels:]

labl_map = [-1] * len(classes)
for i, c in enumerate(classes):
    labl_map[c] = i


print("known_labels:", INLIER)

train_labeled_examples, train_unlabeled_examples = [], []
for example in train_examples:
    if label_list.index(example.label) in INLIER:
        train_labeled_examples.append(example)
   
train_loss = 0


train_labeled_features = convert_examples_to_features(train_labeled_examples, label_list, max_seq_length, tokenizer)
logger.info("***** Running training(labeled) *****")
logger.info("  Num examples = %d", len(train_labeled_features))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
train_labeled_input_ids = torch.tensor([f.input_ids for f in train_labeled_features], dtype=torch.long)
train_labeled_input_mask = torch.tensor([f.input_mask for f in train_labeled_features], dtype=torch.long)
train_labeled_segment_ids = torch.tensor([f.segment_ids for f in train_labeled_features], dtype=torch.long)
train_labeled_label_ids = torch.tensor([labl_map[f.label_id] for f in train_labeled_features], dtype=torch.long)

train_labeled_data = TensorDataset(train_labeled_input_ids, train_labeled_input_mask, train_labeled_segment_ids, train_labeled_label_ids)
train_labeled_sampler = RandomSampler(train_labeled_data)
train_labeled_dataloader = DataLoader(train_labeled_data, sampler=train_labeled_sampler, batch_size=train_batch_size)


## Evaluate for each epcoh
eval_examples = processor.get_dev_examples(data_dir)
eval_labeled_examples = []
for example in eval_examples:
    if label_list.index(example.label) in INLIER:
        eval_labeled_examples.append(example)
        
eval_features = convert_examples_to_features(eval_labeled_examples, label_list, max_seq_length, tokenizer)
logger.info("")
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_features))
logger.info("  Batch size = %d", eval_batch_size)
eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
eval_label_ids = torch.tensor([labl_map[f.label_id] for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)



test_examples = processor.get_test_examples(data_dir)
test_labeled_examples = []
test_outlier_examples = []
for example in eval_examples:
    if label_list.index(example.label) in INLIER:
        test_labeled_examples.append(example)
    else:
        test_outlier_examples.append(example)


test_features = convert_examples_to_features(test_labeled_examples, label_list, max_seq_length, tokenizer)
logger.info("")
logger.info("***** Running testuation *****")
logger.info("  Num examples = %d", len(test_examples))
logger.info("  Batch size = %d", eval_batch_size)
test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
test_label_ids = torch.tensor([labl_map[f.label_id] for f in test_features], dtype=torch.long)
test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

test_features = convert_examples_to_features(test_outlier_examples, label_list, max_seq_length, tokenizer)
logger.info("")
logger.info("***** Running testuation *****")
logger.info("  Num examples = %d", len(test_examples))
logger.info("  Batch size = %d", eval_batch_size)
test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
test_label_ids = torch.tensor([labl_map[f.label_id] for f in test_features], dtype=torch.long)
test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
test_sampler = SequentialSampler(test_data)
openset_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)


def test_classifier(model, eval_dataloader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    total = 0
    total_correct = 0
    y_preds = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits, _  = model(input_ids, segment_ids, input_mask)

        correct = torch.sum(logits.max(dim=1)[1] == label_ids.to(device))
        total += input_ids.size(0)
        total_correct += correct


    accuracy = float(total_correct) / total
    print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
    return accuracy

from sklearn.metrics import roc_curve, roc_auc_score
def plot_roc(known_scores, unknown_scores, do_plot=False, **options):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    auc_score = roc_auc_score(y_true, y_score)

    #fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
    return auc_score


def get_score(preds, mode):
    if mode == 'ce':# 'confidence_threshold':
        return 1 - torch.max(torch.softmax(preds, dim=1), dim=1)[0].data.cpu().numpy()
    elif mode == 'augmented_classifier':
        return torch.softmax(preds, dim=1)[:, -1].data.cpu().numpy()
    elif mode == 'entropy':
        return -((preds * torch.log(preds)).sum(1)).data.cpu().numpy()
    elif mode == 'kliep':#'kliep_threshold':
        return 1-torch.max(preds, dim=1)[0].data.cpu().numpy()
    elif mode == 'gauss':#'kliep_threshold':
        return torch.min(preds, dim=1)[0].data.cpu().numpy()
    assert False

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)


def bpdist(zs,centers):
    #pdist = torch.nn.PairwiseDistance(p=2)
    ret =[]
    for z in zs:
        #ret.append(pdist(z, centers))
        #import pdb; pdb.set_trace()
        ret.append(cos(z.unsqueeze(0), centers))
    return torch.stack(ret)


def test_open_set_performance_gauss(classifier, centers, testing_dataset, openset_dataset, mode):
    # Extracting probabilities Q
    classifier.eval()
    qs = []
    known_scores = []
    known_scoresq = []
    unknown_scores = []
    unknown_scoresq = []
    f1_preds, f1_labels = [], []
    #import pdb; pdb.set_trace()
    thresh=0
    with torch.no_grad():
        for batch in testing_dataset:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            preds, zs  = classifier(input_ids, segment_ids, input_mask)
            known_scores.extend(get_score(preds, mode))
            if centers is not None:
                center_scores = bpdist(zs,centers)
                known_scoresq.extend(get_score(center_scores, 'gauss'))

                for p, lab, cents in zip(preds,label_ids,center_scores):
                    f1_labels.append(lab.item())
                    pval, pind = torch.max(p, dim=0)
                    cval, cind = torch.min(cents, dim=0)
                    if pval > thresh and pind == cind:
                        f1_preds.append(pind.item())
                    else:
                        f1_preds.append(model.num_labels)
            else:
                for p, lab in zip(preds,label_ids):
                    f1_labels.append(lab.item())
                    pval, pind = torch.max(p, dim=0)
                    if pval > thresh:
                        f1_preds.append(pind.item())
                    else:
                        f1_preds.append(model.num_labels)


        for batch in openset_dataset:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            preds, zs  = classifier(input_ids, segment_ids, input_mask)
            unknown_scores.extend(get_score(preds, mode))
            if centers is not None:
                center_scores = bpdist(zs,centers)
                unknown_scoresq.extend(get_score(center_scores, 'gauss'))

                for p, cents in zip(preds, center_scores):
                    f1_labels.append(model.num_labels)
                    pval, pind = torch.max(p, dim=0)
                    cval, cind = torch.min(cents, dim=0)
                    if pval > thresh and pind == cind:
                        f1_preds.append(pind.item())
                    else:
                        f1_preds.append(model.num_labels)
            else:
                for p in preds:
                    f1_labels.append(model.num_labels)
                    pval, pind = torch.max(p, dim=0)
                    if pval > thresh:
                        f1_preds.append(pind.item())
                    else:
                        f1_preds.append(model.num_labels)

    calc_f1(f1_preds, f1_labels)

    auc = plot_roc(known_scores, unknown_scores, mode)
    print('Detecting with mode {}, avg. known-class score: {:.03f}, avg unknown score: {:.03f}'.format(
        mode, np.mean(known_scores), np.mean(unknown_scores)))
    print('Mode {}: generated ROC with AUC score {:.04f}'.format(mode, auc))
    
    if centers is not None:
        aucq = plot_roc(known_scoresq, unknown_scoresq, mode)
        print('Detecting with mode {}, avg. known-class score: {:.03f}, avg unknown score: {:.03f}'.format(
            "gauss", np.mean(known_scoresq), np.mean(unknown_scoresq)))
        print('Mode {}: generated ROC with AUC score {:.04f}'.format("gauss", aucq))
        auc = max(auc, aucq)
    return auc


def get_centers(model, train_loader, num_classes, device):

    with torch.no_grad():
        zs = {}
        for i in range(num_classes): zs[i] = torch.empty(0).to(device)

        for batch in tqdm(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits, z = model(input_ids, input_mask, segment_ids)

            for i in range(logits.size(1)):
                lbl_idx = label_ids == i
                if lbl_idx.sum() == 0: continue

                zs[i] = torch.cat([zs[i], z[lbl_idx]])


        #centers = torch.zeros(num_classes, 2048).to(device)
        centers = torch.zeros(num_classes, z.size(1)).to(device)
        for i in range(num_classes):
            centers[i] = zs[i].mean(0)
    return centers

def calc_f1(preds, y_true):
    preds  = np.array(preds)
    y_true = np.array(y_true)
    f1_micro  = f1_score(y_true=y_true, y_pred=preds, average="micro")
    f1_macro  = f1_score(y_true=y_true, y_pred=preds, average="macro")
    f1_weight = f1_score(y_true=y_true, y_pred=preds, average="weighted")

    acc = (preds == y_true).mean()

    print(f"F-measures: macro {f1_macro:.4f}, micro {f1_micro:.4f}, weighted {f1_weight:.4f}")


global_step = 0
criterion = torch.nn.CrossEntropyLoss()
best_auc = 0

y_pred_last = np.zeros_like(eval_label_ids)
for epoch in trange(int(num_train_epochs), desc="Epoch"):
    model.train()
    do_centers = epoch > -2 and True
    if do_centers:
        print("getting centers")
        centers = get_centers(model, train_labeled_dataloader, num_known_labels, device)
    else:
        centers = None


    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_labeled_dataloader, desc="Iteration (labeled)")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, z  = model(input_ids, segment_ids, input_mask)
        
        #loss = torch.zeros(0, requires_grad=True).to(device).sum()
        if loss_type.startswith('kliep'):
            preds = torch.sigmoid(logits)*10
              
            loss = None
            logit_count = preds.size(1)
            for i in range(logit_count):
                lbl_idx = label_ids == i
                if lbl_idx.sum() == 0: continue
                logit_index = torch.tensor([x for x in range(preds.size(1)) if x!=i ])

                lloss = lam * (1/logit_count) * (preds[lbl_idx][:,logit_index].mean(1) -torch.log(preds[lbl_idx][:,i])).mean()
                if loss is None:
                    loss = lloss
                else:
                    loss+= lloss
        elif loss_type.startswith('ce'):
            loss = lam * criterion(logits, label_ids)
        elif loss_type != 'original':
            assert False

        if do_centers:
            loss+=cos(z, centers[label_ids]).mean()

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    train_labeled_loss = tr_loss / nb_tr_steps

    if epoch > 3:
        test_classifier(model, test_dataloader)
        cur_auc  = test_open_set_performance_gauss(model, centers, test_dataloader, openset_dataloader, loss_type)
        
        if cur_auc > best_auc:
            best_auc = cur_auc

print(f"best_auc: {best_auc}") 
   
exit("sucessfully trained")
#plot_confusion_matrix(cm, label_list, normalize=False, figsize=(8, 8),
#                      title='Confusion matrix, accuracy=' + str(results['ACC']))
# Save a trained model and the associated configuration
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
torch.save(model_to_save.state_dict(), output_model_file)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
with open(output_config_file, 'w') as f:
    f.write(model_to_save.config.to_json_string())
    


results = clustering_score(y_true, y_pred)
results_all.update({'CDAC+': results})
print("final results")
print(results)
with open(f"results.csv", "a") as writer:
    nmi = results["NMI"]
    ari = results["ARI"]
    acc = results["ACC"]
    writer.write(f"{seed}, {task_name}, {fraction}, {labeled_ratio}, {unknown_cls_ratio}, {loss_type}, CDAC+  , {nmi}, {ari}, {acc} \n")



print("all results?")
print(results_all)
