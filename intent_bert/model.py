import torch
from torch import nn
from torch.nn import Parameter
from typing import Optional
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch.nn.functional as F



class BertForConstrainClustering(BertPreTrainedModel):
    def __init__(self, config, num_labels, last_layer):
        super(BertForConstrainClustering, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        
        # train
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Pooling-mean
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fc = last_layer(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        
        # finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(num_labels, num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        eps = 1e-10
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1)) # Pooling-mean
        self.z = pooled_output
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)
