from utils import *
import argparse
import random
import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import trange
from tqdm import tqdm
from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel

import torch.nn.functional as F

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

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
train = False

max_seq_length = max_seq_task[task_name]
train_batch_size = 512 if train else 24
eval_batch_size= 2048 if train else 24
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
num_known_labels = 6
print(f"total labels: {num_labels}, known_labels: {num_known_labels}")
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        self.z = pooled_output
        return logits

# Freezing all transformer (except the last layer)
model = BertForSequenceClassification.from_pretrained(bert_model, num_labels = num_known_labels)
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
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=(1 / num_train_epochs), num_training_steps=num_train_optimization_steps
)

## Settings of unknown classes discovery
# 1. Select 25% classes as unknown(-1)
# 2. Set 90% of examples as unknown(-1)

classes = list(range(0,num_labels))
while True:
    random.shuffle(classes)
    INLIER = classes[:num_known_labels]
    OUTLIER = classes[num_known_labels:]
    if OUTLIER[0] == 1:
        break

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
            logits  = model(input_ids, segment_ids, input_mask)

        correct = torch.sum(logits.max(dim=1)[1] == label_ids.to(device))
        total += input_ids.size(0)
        total_correct += correct


    accuracy = float(total_correct) / total
    print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
    return accuracy
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
def save_data(model, dataloader, file_base):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    total = 0
    total_correct = 0
    y_preds = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits, _  = model(input_ids, segment_ids, input_mask)

        correct = torch.sum(logits.max(dim=1)[1] == label_ids.to(device))
        total += input_ids.size(0)
        total_correct += correct



    

def save_data_forviz(model, base, dataloader):
    model.eval()

    lig = LayerIntegratedGradients(model, model.bert.embeddings)

    preds = []
    labels = []
    words = []
    sentences = []
    zs = torch.empty(0)
    new_label_list = []


    for step, batch in enumerate(tqdm(dataloader, desc="parsing data: "+base)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits = model(input_ids, input_mask, segment_ids, labels=label_ids)
        ps  = torch.softmax(logits, dim=1)

        ref_input_ids = input_ids.clone()
        ref_input_ids[:,1:] = tokenizer.pad_token_id
        sep_ind = input_ids == tokenizer.sep_token_id
        ref_input_ids[sep_ind] = tokenizer.sep_token_id
        attributions, delta = lig.attribute(inputs=input_ids,baselines=ref_input_ids, target=torch.argmax(logits, dim = 1), additional_forward_args=(input_mask, segment_ids), return_convergence_delta=True)
        attributions_sum = summarize_attributions(attributions)
        
        #print("integrated gradient values", attributions_sum)
        idx_important_word = torch.argmax(attributions_sum, dim = 1)
        for i, (indx, sent) in enumerate(zip(idx_important_word, input_ids)):
            words.append(tokenizer.decode(sent[indx].item()).replace(" ", ""))
            sent = sent.detach().cpu().numpy()[1:torch.nonzero(sep_ind[i]).item()]
            sentences.append(tokenizer.decode(sent))

        zs = torch.cat([zs, model.z.detach().cpu()], dim = 0)
        #preds.extend(ps.detach().cpu().numpy()) # these need to be sorted according to true label

        labels.extend([label_list[labl_map.index(lab.item())] for lab in label_ids])


    np.savetxt(base + "_zs.txt",         zs.numpy())
    #np.savetxt(base + "_preds.txt",      np.array(preds))
    np.savetxt(base + "_words.txt",      np.array(words), fmt="%s")
    np.savetxt(base + '_sentences.txt',  np.array(sentences), fmt="%s")
    np.savetxt(base + '_labels.txt',     np.array(labels), fmt="%s")

if train:
    global_step = 0
    criterion = torch.nn.CrossEntropyLoss()
    best_auc = 0

    y_pred_last = np.zeros_like(eval_label_ids)
    for epoch in trange(int(num_train_epochs), desc="Epoch"):
        model.train()

        for step, batch in enumerate(tqdm(train_labeled_dataloader, desc="Iteration (labeled)")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, input_mask, segment_ids, labels=label_ids)
            loss = criterion(logits, label_ids)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    torch.save(model.state_dict(), "bert.pth")
else:
    model.load_state_dict(torch.load("bert.pth"))
    save_data_forviz(model, "test", test_dataloader)
    save_data_forviz(model, "outlier", openset_dataloader)
    save_data_forviz(model, "train", train_labeled_dataloader)

#logits = model(input_ids, input_mask, segment_ids, labels=label_ids)



   
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
