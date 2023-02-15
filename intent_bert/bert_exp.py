from model import BertForConstrainClustering
from bertutils import *
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
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from datetime import datetime
import warnings
import copy

from utils import CustomLayers, save_openset_all, Losses

warnings.warn = warn

results_all = {}
seed = 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


task_name = dataset = sys.argv[1]
fraction = float(sys.argv[2])
loss_type = sys.argv[3] #kliep, ce, original
f_layer = sys.argv[4]
seed = int(sys.argv[5])
full_task_name = "_".join(sys.argv[1:])

file_base      = f"{loss_type}_{f_layer}_{task_name}_{fraction}"
out_file_last  = os.path.join("results","last", file_base + '.csv')
out_file_val   = os.path.join("results","val" , file_base + '.csv')

print("Starting task of: " + full_task_name)
data_dir = 'data/' + task_name
output_dir = 'outmod/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
bert_model = "bert-base-uncased"#"/data/disk1/sharing/pretrained_embedding/bert/uncased_L-12_H-768_A-12"
num_train_epochs = 30
 
max_seq_task = {
    "snips": 35,
    'dbpedia': 54,
    "stackoverflow": 20,
    "20ng": 64,
}
max_seq_length = max_seq_task[task_name]
train_batch_size = 256
eval_batch_size= 512
learning_rate = 1e-3
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
print("device: {} n_gpu: {}".format(device, n_gpu))
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
last_layer = CustomLayers[f_layer]
modes = last_layer.get_modes()

model = BertForConstrainClustering.from_pretrained(bert_model, num_known_labels, last_layer)
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
for example in tqdm(train_examples):
    if label_list.index(example.label) in INLIER:
        train_labeled_examples.append(example)
   
train_loss = 0


train_labeled_features = convert_examples_to_features(train_labeled_examples, label_list, max_seq_length, tokenizer)
print("***** Running training(labeled) *****")
print("  Num examples = %d", len(train_labeled_features))
print("  Batch size = %d", train_batch_size)
print("  Num steps = %d", num_train_optimization_steps)
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
print("")
print("***** Running evaluation *****")
print("  Num examples = %d", len(eval_features))
print("  Batch size = %d", eval_batch_size)
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
print("")
print("***** Running testuation *****")
print("  Num examples = %d", len(test_examples))
print("  Batch size = %d", eval_batch_size)
test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
test_label_ids = torch.tensor([labl_map[f.label_id] for f in test_features], dtype=torch.long)
test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

test_features = convert_examples_to_features(test_outlier_examples, label_list, max_seq_length, tokenizer)
print("")
print("***** Running testuation *****")
print("  Num examples = %d", len(test_examples))
print("  Batch size = %d", eval_batch_size)
test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
test_label_ids = torch.tensor([labl_map[f.label_id] for f in test_features], dtype=torch.long)
test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
test_sampler = SequentialSampler(test_data)
openset_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

@torch.no_grad()
def test_classifier(model, cur_dataset):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    total = 0
    total_correct = 0
    y_preds = []
    for batch in tqdm(cur_dataset, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        output = model(input_ids, segment_ids, input_mask)
        if type(output) is tuple:
            output = output[0]

        correct = torch.sum(output.max(dim=1)[1] == label_ids.to(device))
        total += input_ids.size(0)
        total_correct += correct


    accuracy = float(total_correct) / total
    print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
    return accuracy





global_step = 0
my_losses = Losses()
criterion = my_losses.get_loss_dict()[loss_type]
best_acc = 0

y_pred_last = np.zeros_like(eval_label_ids)
for epoch in trange(int(num_train_epochs), desc="Epoch"):
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_labeled_dataloader, desc="Iteration (labeled)")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits = model(input_ids, segment_ids, input_mask)
        
        
        loss = criterion(logits, label_ids)

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    train_labeled_loss = tr_loss / nb_tr_steps
    cur_acc = test_classifier(model, eval_dataloader)
    if cur_acc > best_acc: 
        best_acc = cur_acc
        best_val_model = copy.deepcopy(model.state_dict())
    #test_open_set_performance_gauss(model, test_dataloader, openset_dataloader, loss_type)
    


save_openset_all(model, test_dataloader, openset_dataloader, modes, out_file_last, seed)
model.load_state_dict(best_val_model)
save_openset_all(model, test_dataloader, openset_dataloader, modes, out_file_val, seed)
