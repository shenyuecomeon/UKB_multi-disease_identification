import torch
import torch.nn as nn
import ipdb
import math
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel, AutoModel, BertPreTrainedModel, BertConfig, AutoTokenizer
import pandas as pd

class POPDxModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True)])

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
        x = torch.relu(x) 
        x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))        
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class NN(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(NN, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear( y_emb.shape[1],label_num,bias=True)])
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)  
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
class LGclassifier(nn.Module):
    def __init__(self, feature_size, nlabel):
        super(LGclassifier, self).__init__()
        self.main = nn.Sequential(           
            nn.Linear(feature_size, nlabel)
        )

    def forward(self, input):
        return self.main(input)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)



class LGclassifier(nn.Module):
    def __init__(self, feature_size, nlabel):
        super(LGclassifier, self).__init__()
        self.main = nn.Sequential(           
            nn.Linear(feature_size, nlabel)
        )

    def forward(self, input):
        return self.main(input)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class PrefixEncoder(torch.nn.Module):
    def __init__(self, config, pre_seq_len):
        super().__init__()
        '''
        The torch.nn model to encode the prefix

        Input shape: (batch-size, prefix-length)

        Output shape: (batch-size, prefix-length, 2*layers*hidden)
        '''

        self.embedding = torch.nn.Embedding(pre_seq_len,  config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )

    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values



class BertPrefixForMLclassification_conv(BertPreTrainedModel):
    def __init__(self,config,feature_num,hidden_size,y_emb,pre_seq_len):
        super(BertPrefixForMLclassification_conv, self).__init__(config)
        self.pre_seq_len = pre_seq_len
        #self.bert = BertModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc", add_pooling_layer=False)
        self.bert = BertModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc", output_hidden_states=True, return_dict=True)
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, self.pre_seq_len)
        for param in self.bert.parameters():
            param.requires_grad=False # NOTE:BERT参数不变化
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num+768, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True)])

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
                )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        #(n_layer, bs,n_head, seq_len, n_embed)
        #(2, 16, 12, 50, 64)
        # ipdb.set_trace()
        return past_key_values

    def forward(self, x, input_ids, attention_mask):
        # ipdb.set_trace()
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        outputs = self.bert(input_ids = input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
        # bert_params = list(self.bert.parameters())
        # prefix_params = list(self.prefix_encoder.parameters())

        text_embed = outputs[0][:,0,:] #(bs,768)
        # text_embed = outputs.pooler_output
        x = torch.concat([x, text_embed],dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i ==0:
                x = torch.relu(x) 
        # ipdb.set_trace()
        x_ = torch.matmul(x, torch.transpose(self.y_emb,0,1))  
        # ipdb.set_trace()
        # for name, param in self.bert.named_parameters():
        #     print(f"{name}: {param.requires_grad}")
        
        # return x,x_
        return x_
        
    def initialize(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
