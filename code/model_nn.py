import torch
import torch.nn as nn
import ipdb
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel, AutoModel, BertPreTrainedModel, BertConfig, AutoTokenizer



class nn_twolayer(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(nn_twolayer, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  label_num, bias=True)])
 
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

class nn_fourlayer(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(nn_fourlayer, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size,  label_num, bias=True)])
 
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

class POPDx_deeplift_nn(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_deeplift_nn, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num+768, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, 1268, bias=True),
                                      nn.Linear(1268, y_emb.shape[1], bias=True)])

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            if i ==0:
                x = torch.relu(x)  
            x = linear(x)
        x = torch.matmul(x, self.y_emb)        
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDxModel_nn_conv(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDxModel_nn_conv, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  1268, bias=True),
                                      nn.Linear( 1268,label_num,bias=True)])
        # self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
        #                               nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
        #                               nn.Linear( y_emb.shape[1],label_num,bias=True)])
        # self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
        #                               nn.Linear(hidden_size,2*hidden_size,bias=True),
        #                               nn.Linear(2*hidden_size,4*hidden_size,bias=True),
        #                               nn.Linear(4*hidden_size, y_emb.shape[1],bias=True)])

    # def forward(self, x):
        # x = self.embedding(x)
        # for (i, linear) in enumerate(self.linears):
        #     x = linear(x)
        #     if i < 3:
        #         x = torch.relu(x)  
        # x = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))  
        # return x
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)  
        x = torch.matmul(x,self.y_emb)
        return x
    # def forward(self, x):
    #     for (i, linear) in enumerate(self.linears):
    #         x = linear(x)
    #         if i == 1:
    #             x = torch.relu(x)  
        # return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDx_notext_mlp(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_notext_mlp, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear( y_emb.shape[1],label_num,bias=True)])

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 3:
                x = torch.relu(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDx_text(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_text, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear(y_emb.shape[1], label_num,bias=True)])

    def forward(self, x, text):
        x = torch.concat([x,text],dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)  
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)     

class POPDx_text_mlp(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, text_num, y_emb):
        super(POPDx_text_mlp, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(text_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, text_num, bias=True)])
        self.linears1 = nn.ModuleList([nn.Linear(feature_num + text_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True),
                                      nn.Linear(y_emb.shape[1], label_num,bias=True)])
        
    def forward(self, x, text):
        for (i, linear) in enumerate(self.linears):
            text = linear(text)
        text_embed = torch.relu(text)

        x = torch.concat([x, text_embed], dim=1)
        for (i, linear) in enumerate(self.linears1):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)        
        return x

class POPDx_all_mlp(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_all_mlp, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size, hidden_size, bias=True),
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear(y_emb.shape[1], label_num,bias=True)])

    def forward(self, x, text):
        x = torch.concat([x,text],dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 3:
                x = torch.relu(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDx_onlytext(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_onlytext, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb

        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear( y_emb.shape[1],label_num,bias=True)])

    def forward(self, a, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDx_onlytext_mlp(nn.Module):
    def __init__(self, label_num, hidden_size, text_num, y_emb):
        super(POPDx_onlytext_mlp, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(text_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, text_num, bias=True)])
        self.linears1 = nn.ModuleList([nn.Linear(text_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True),
                                      nn.Linear(y_emb.shape[1], label_num,bias=True)])
    def forward(self, a, text):
        for (i, linear) in enumerate(self.linears):
            text = linear(text)
        text_embed = torch.relu(text)

        x = text_embed
        for (i, linear) in enumerate(self.linears1):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class POPDxLoRAModel(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb, r):
        super(POPDxLoRAModel, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.r = r
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True),
                                      nn.Linear(hidden_size, y_emb.shape[1], bias=True)])
        self.lora_A = nn.Parameter(torch.empty(y_emb.shape[1], r))
        self.lora_B = nn.Parameter(torch.empty(r, label_num))
        self.scaling = 1 / self.r

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
        x = torch.relu(x)
        x_lora = (self.lora_A @ self.lora_B) * self.scaling
        x_result = torch.matmul(x, torch.transpose(self.y_emb, 0, 1))
        x_result += torch.matmul(x, x_lora)

        return x_result, None

    def initialize(self):
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class LoRAModel(nn.Module):
    def __init__(self, label_num):
        super(LoRAModel, self).__init__()
        self.biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
        self.linear = nn.Linear(self.bert.config.hidden_size, label_num)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.linear(pooled_output)
        return logits


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

       
class POPDx_text_bert_convnn(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_text_bert_convnn, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  1268, bias=True),
                                      nn.Linear( 1268,label_num,bias=True)])
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    def forward(self, x, input_ids, attention_mask):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)['last_hidden_state']
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        text_embed= torch.div(text_embed,padd_len)

        x = torch.concat([x, text_embed], dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 0:
                x = torch.relu(x)  
        x = torch.matmul(x,self.y_emb)
        return x
    
    def initialize(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)   

class POPDx_text_bert_conv(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_text_bert_conv, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size, label_num, bias=True)])
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    def forward(self, x, input_ids, attention_mask):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)['last_hidden_state']
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        text_embed= torch.div(text_embed,padd_len)

        x = torch.concat([x, text_embed], dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 0:
                x = torch.relu(x)  
        x = torch.matmul(x,self.y_emb)
        return x
    
    def initialize(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  

class POPDx_text_bert(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, y_emb):
        super(POPDx_text_bert, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.linears = nn.ModuleList([nn.Linear(feature_num, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear( y_emb.shape[1],label_num,bias=True)])
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    def forward(self, x, input_ids, attention_mask):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)['last_hidden_state']
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        text_embed= torch.div(text_embed,padd_len)

        x = torch.concat([x, text_embed], dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 0:
                x = torch.relu(x)  
        return x
    
    def initialize(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)    
         

class POPDx_text_scale(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size, bert_hidden_num, y_emb):
        super(POPDx_text_scale, self).__init__()
        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.y_emb = y_emb
        self.scale = nn.Linear(feature_num, bert_hidden_num)
        self.linears = nn.ModuleList([nn.Linear(bert_hidden_num*2, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear( y_emb.shape[1],label_num,bias=True)])
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    def forward(self, x, input_ids, attention_mask):
        # x = self.embedding(x)
        # text.requires_grad_()
        x = self.scale(x)
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)['last_hidden_state']
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        text_embed= torch.div(text_embed,padd_len)

        x = torch.concat([x, text_embed], dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)  
        return x
    
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


class BertPrefixForMLclassification(BertPreTrainedModel):
    def __init__(self,config,feature_num,hidden_size,y_emb,label_num,pre_seq_len):
        super(BertPrefixForMLclassification, self).__init__(config)
        self.pre_seq_len = pre_seq_len
        #self.bert = BertModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc", add_pooling_layer=False)
        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc", output_hidden_states=True, return_dict=True)
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, self.pre_seq_len)
        for param in self.bert.parameters():
            param.requires_grad=False # NOTE:BERT参数不变化

        self.linears = nn.ModuleList([nn.Linear(feature_num+768, hidden_size, bias=True), 
                                      nn.Linear(hidden_size,  y_emb.shape[1], bias=True),
                                      nn.Linear(y_emb.shape[1], label_num,bias=True)])
        

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
        #text_embed = outputs.pooler_output
        x = torch.concat([x, text_embed],dim=1)
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            if i == 1:
                x = torch.relu(x)  
        # ipdb.set_trace()
        # for name, param in self.bert.named_parameters():
        #     print(f"{name}: {param.requires_grad}")
        
        return x
        
    
    def initialize(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class BertPrefixForMLcls_onlytext(BertPreTrainedModel):
    def __init__(self,config,hidden_size,y_emb,label_num,pre_seq_len):
        super(BertPrefixForMLcls_onlytext, self).__init__(config)
        self.pre_seq_len = pre_seq_len
        self.bert =BertModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc", add_pooling_layer=False)
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, self.pre_seq_len)
        for param in self.bert.parameters():
            param.requires_grad=False # NOTE:BERT参数不变化

        self.linear = nn.Linear(768, label_num, bias=True)
        

    def get_prompt(self, batch_size, device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
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

    def forward(self, input_ids, attention_mask, device):

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size, device = device)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        outputs = self.bert(input_ids = input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
        # bert_params = list(self.bert.parameters())
        # prefix_params = list(self.prefix_encoder.parameters())
        text_embed = outputs[0][:,0,:] #(bs,768)
        x = self.linear(text_embed)
        x = torch.relu(x)  

        # ipdb.set_trace()
        # for name, param in self.bert.named_parameters():
        #     print(f"{name}: {param.requires_grad}")
        return x
        
    def initialize(self):
        nn.init.kaiming_normal_(self.linear.weight)  