import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import pandas as pd
import numpy as np

from sklearn.utils.graph import graph_shortest_path
from scipy.sparse.linalg import svds
import json 
import time
from transformers import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from transformers import AutoTokenizer, AutoModel
import torch
import ipdb
from data import Dataset_embed
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from transformers import LongformerModel, LongformerTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def svd_emb(mat, dim=20):
    U, S, V = svds(mat, k=dim)
    X = np.dot(U, np.sqrt(np.diag(S)))
    return X,S

def svd_new(mat):
    U, S, V = np.linalg.svd(mat)
    energy = np.square(S)
    total_energy = np.sum(energy)
    energy_ratio = energy/total_energy
    cumulative_energy_ratio = np.cumsum(energy_ratio)
    ipdb.set_trace()
    dim = np.argmax(cumulative_energy_ratio >= 0.9) + 1

def ontology_emb(dim=500, ICD_network_file = '../data/19_rearrange.csv', save_dir = './rearrange_embeddings/', use_pretrain = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if use_pretrain:
        try:
            with open(os.path.join(save_dir,'embedding_node_to_idx_dict.json'), 'r') as inpt:
                l2i = json.load(inpt)
            with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'r') as inpt:
                i2l = json.load(inpt)
            # sp = np.load(os.path.join(save_dir, 'sp.npy'))
            # svd_new(sp)
            # X_emb, S = svd_emb(sp, dim=dim)
            # np.save("./icd_embeddings/icd_svd_dim250.npy",X_emb)
            return (l2i, i2l, X_emb)
        except:
            err = 'No pretrained ONTOLOGY embeddings found.'
            print(err)
    
    df_network=pd.read_csv(ICD_network_file, delimiter=',')
    df_network.node_id = df_network.node_id.astype("int")
    df_network.parent_id = df_network.parent_id.astype("int")


    s2p = {}
    lset = set()
    for i,row in df_network.iterrows():

        s = row.coding
        if row.parent_id==0:
            p='ROOT'
        else:
            
            p = df_network.loc[df_network['node_id']==row.parent_id, 'coding'].values.item()
        wt = 1.
        
        if s not in s2p:
            s2p[s] = {}
        s2p[s][p] = wt
        lset.add(s)
        lset.add(p)
    ipdb.set_trace()
    
    lset = np.sort(list(lset))
    nl = len(lset)
    l2i = dict(zip(lset, range(nl)))
    i2l = dict(zip(range(nl), lset))

    embeddings_to_index_dict = dict(zip(lset,range(len(lset))))
    index_to_embeddings_dict = dict(zip(range(len(lset)),lset))

    with open(os.path.join(save_dir, 'icd_embedding_node_to_idx_dict.json'), 'w') as output:
        json.dump(embeddings_to_index_dict, output)
    with open(os.path.join(save_dir, 'icd_idx_to_embedding_node_dict.json'), 'w') as output:
        json.dump(index_to_embeddings_dict, output)

    print('Num of relationships:',len(s2p))
    print('Num edges:',nl)

    A = np.zeros((nl, nl))
    for s in s2p:
        for p in s2p[s]:
            A[l2i[s], l2i[p]] = s2p[s][p]
            A[l2i[p], l2i[s]] = s2p[s][p]
    
    time0=time.time()
    ipdb.set_trace()
    sp = graph_shortest_path(A,method='FW',directed=False)

    print(time.time()-time0)
    # X_emb, S = svd_emb(sp, dim=dim)
    np.save(os.path.join(save_dir, 'sp.npy'), sp)
    X_emb, S = svd_emb(sp, dim=dim)
    #U:ndarray, shape=(M, k)
    #S:ndarray, shape=(k,)
    #V:ndarray, shape=(k, N)
    sp *= -1.

    np.save(os.path.join(save_dir, 'icd_svd_dim250.npy'), X_emb)   
    return (l2i, i2l, X_emb)

def ontology_emb_rearrange(dim=500, ICD_network_file = '../data/19_rearrange.csv', save_dir = './rearrange_embeddings/', use_pretrain = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if use_pretrain:
        try:
            with open(os.path.join(save_dir,'embedding_node_to_idx_dict.json'), 'r') as inpt:
                l2i = json.load(inpt)
            with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'r') as inpt:
                i2l = json.load(inpt)
            # sp = np.load(os.path.join(save_dir, 'sp.npy'))
            # svd_new(sp)
            # X_emb, S = svd_emb(sp, dim=dim)
            # np.save("./icd_embeddings/icd_svd_dim250.npy",X_emb)
            return (l2i, i2l, X_emb)
        except:
            err = 'No pretrained ONTOLOGY embeddings found.'
            print(err)
    
    df_network=pd.read_csv(ICD_network_file, delimiter=',')
    df_network.node_id = df_network.node_id.astype("int")
    df_network.parent_id = df_network.parent_id.astype("int")


    s2p = {}
    lset = ['ROOT']
    for i,row in df_network.iterrows():

        s = row.coding
        if row.parent_id==0:
            p='ROOT'
        else:
            
            p = df_network.loc[df_network['node_id']==row.parent_id, 'coding'].values.item()
        wt = 1.
        
        if s not in s2p:
            s2p[s] = {}
        s2p[s][p] = wt
        lset.append(s)
    ipdb.set_trace()
    
    lset = np.array(lset)
    nl = len(lset)
    l2i = dict(zip(lset, range(nl)))
    i2l = dict(zip(range(nl), lset))

    embeddings_to_index_dict = dict(zip(lset,range(len(lset))))
    index_to_embeddings_dict = dict(zip(range(len(lset)),lset))

    with open(os.path.join(save_dir, 'icd_embedding_node_to_idx_dict.json'), 'w') as output:
        json.dump(embeddings_to_index_dict, output)
    with open(os.path.join(save_dir, 'icd_idx_to_embedding_node_dict.json'), 'w') as output:
        json.dump(index_to_embeddings_dict, output)

    print('Num of relationships:',len(s2p))
    print('Num edges:',nl)

    A = np.zeros((nl, nl))
    for s in s2p:
        for p in s2p[s]:
            A[l2i[s], l2i[p]] = s2p[s][p]
            A[l2i[p], l2i[s]] = s2p[s][p]
    
    time0=time.time()
    ipdb.set_trace()
    sp = graph_shortest_path(A,method='FW',directed=False)

    print(time.time()-time0)
    # X_emb, S = svd_emb(sp, dim=dim)
    np.save(os.path.join(save_dir, 'sp.npy'), sp)
    X_emb, S = svd_emb(sp, dim=dim)
    #U:ndarray, shape=(M, k)
    #S:ndarray, shape=(k,)
    #V:ndarray, shape=(k, N)
    sp *= -1.

    np.save(os.path.join(save_dir, 'icd_svd_dim250.npy'), X_emb)   
    return (l2i, i2l, X_emb)
def ontology_emb_ICD_phe(dim=500, 
                         ICD_network_file1 = '../data/19.csv',
                         ICD_network_file2="../data/20_phecode.csv",
                         ICD_network_file3="../data/Phecode_map_icd10.csv",
                         save_dir = './embeddings/', use_pretrain = True):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    df_network1 = pd.read_csv(ICD_network_file1, delimiter=',')
    df_network2 = pd.read_csv(ICD_network_file2)
    df_network2 = pd.read_csv(ICD_network_file3)

    s2p = {}
    lset = set()

    for i,row in df_network1.iterrows():

        s = row.coding
        if row.parent_id==0:
            p='ROOT'
        else:
            p = df_network1.loc[df_network1['node_id']==row.parent_id, 'coding'].values.item()
        wt = 1.
        if s not in s2p:
            s2p[s] = {}
        s2p[s][p] = wt
        lset.add(s)
        lset.add(p)

    ipdb.set_trace()
    lset = np.sort(list(lset))
    nl = len(lset)
    l2i = dict(zip(lset, range(nl)))
    i2l = dict(zip(range(nl), lset))

    embeddings_to_index_dict = dict(zip(lset,range(len(lset))))
    index_to_embeddings_dict = dict(zip(range(len(lset)),lset))

    with open(os.path.join(save_dir, 'embedding_node_to_idx_dict.json'), 'w') as output:
        json.dump(embeddings_to_index_dict, output)
    with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'w') as output:
        json.dump(index_to_embeddings_dict, output)

    print('Num of relationships:',len(s2p))
    print('Num edges:',nl)

    A = np.zeros((nl, nl))
    for s in s2p:
        for p in s2p[s]:
            A[l2i[s], l2i[p]] = s2p[s][p]
            A[l2i[p], l2i[s]] = s2p[s][p]
    
    time0=time.time()
    sp = graph_shortest_path(A,method='FW',directed=False)
    ipdb.set_trace()
    print(time.time()-time0)
    # X_emb, S = svd_emb(sp, dim=dim)
    X_emb, S = svd_emb(sp, dim=dim)
    #U:ndarray, shape=(M, k)
    #S:ndarray, shape=(k,)
    #V:ndarray, shape=(k, N)
    sp *= -1.

    np.save(os.path.join(save_dir, 'onto_SVD_embedding_dim.npy'), X_emb)   
    return (l2i, i2l, X_emb)


def phecode_ontoemb(dim=500, ICD_network_file = '../data/20_rearrange.csv', save_dir = './rearranges_embeddings/', use_pretrain = True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if use_pretrain:
        try:
            with open(os.path.join(save_dir,'phe_embedding_node_to_idx_dict.json'), 'r') as inpt:
                l2i = json.load(inpt)
            with open(os.path.join(save_dir, 'phe_idx_to_embedding_node_dict.json'), 'r') as inpt:
                i2l = json.load(inpt)
            return (l2i, i2l, X_emb)
        except:
            err = 'No pretrained ONTOLOGY embeddings found.'
            print(err)
    
    df_network=pd.read_csv(ICD_network_file, delimiter=',')

    s2p = {}
    lset = []
    for i,row in df_network.iterrows():

        s = row.phecode
        if row.node_id==2115:
            p='ROOT'
        else:
            p = df_network.loc[df_network['node_id']==row.parent_id, 'phecode'].values.item()
        wt = 1.
        if s not in s2p:
            s2p[s] = {}
        s2p[s][p] = wt
        lset.append(s)


    # ipdb.set_trace()
    lset = np.array(lset)
    nl = len(lset)
    l2i = dict(zip(lset, range(nl)))
    i2l = dict(zip(range(nl), lset))

    embeddings_to_index_dict = dict(zip(lset,range(len(lset))))
    index_to_embeddings_dict = dict(zip(range(len(lset)),lset))

    with open(os.path.join(save_dir, 'embedding_node_to_idx_dict.json'), 'w') as output:
        json.dump(embeddings_to_index_dict, output)
    with open(os.path.join(save_dir, 'idx_to_embedding_node_dict.json'), 'w') as output:
        json.dump(index_to_embeddings_dict, output)

    print('Num of relationships:',len(s2p))
    print('Num edges:',nl)

    A = np.zeros((nl, nl))
    for s in s2p:
        for p in s2p[s]:
            A[l2i[s], l2i[p]] = s2p[s][p]
            A[l2i[p], l2i[s]] = s2p[s][p]
    np.save("./rearrange_embeddings/phe_adj_arrange.npy",A)
    ipdb.set_trace()
    time0=time.time()
    sp = graph_shortest_path(A,method='FW',directed=False)
    svd_new(sp)
    
    print(time.time()-time0)
    # # X_emb, S = svd_emb(sp, dim=dim)
    
    X_emb, S = svd_emb(sp, dim=dim)
    # #U:ndarray, shape=(M, k)
    # #S:ndarray, shape=(k,)
    # #V:ndarray, shape=(k, N)
    # sp *= -1.

    np.save(os.path.join(save_dir, 'SVD_embedding_dim'+str(dim)+'.npy'), X_emb)   
    return (l2i, i2l, X_emb)


def df_to_batch(df, batch_size=64):
    batch_dict = {}
    batch_s = list(range(len(df)))[::batch_size]
    batch_ = list(zip(batch_s,batch_s[1:]+[len(df)+1]))
    batch_i = 0
    for start_i, end_i in batch_:
        batch_i += 1
        batch_dict[batch_i] = df.iloc[start_i:end_i,:]                  
    return batch_dict

def biobert_embed(df, model, tokenizer, device):
    
    tokenized = df['meaning'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 500
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    ipdb.set_trace()
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    
    input_ids = torch.tensor(padded).to(device)  
    attention_mask = torch.tensor(attention_mask).to(device)
    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)
    # ipdb.set_trace()
    return df['coding'].tolist(), hidden_states,attention_mask

def biobert_embed_text(df, model, tokenizer, device):
    # ipdb.set_trace()
    tokenized = df.iloc[:,0].apply((lambda x: tokenizer.encode(x,  add_special_tokens = False, truncation=True ,max_length =512)))
    max_len = 512

    # for i in tokenized.values:
    #     if len(i) > max_len:
    #         max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask).to(device)
    input_ids = torch.tensor(padded).to(device)  
    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)
    
    return hidden_states, attention_mask

def biobert_embed_phecode(df, model, tokenizer, device):
    # ipdb.set_trace()
    tokenized = df['phenotype'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 500
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    
    input_ids = torch.tensor(padded).to(device)  
    attention_mask = torch.tensor(attention_mask).to(device)
    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)
    phecode_list = ["{:0.2f}".format(float(x)) for x in df['phecode'].tolist()] 
    return phecode_list, hidden_states,attention_mask


def run_bert(batch_size=128, use_pretrain = True):
    # if use_pretrain:
    #     try:
    #         biboert_embeddings_dict = dict(np.load(os.path.join('./embeddings/biobert_embeddings_dict.npz')))
    #         features = np.load(os.path.join('./embeddings/biobert_embeddings.npy'))
    #         return features, biboert_embeddings_dict
    #     except:
    #         print('No pretrained BERT embeddings found.')

    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"  
    print(device)
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    # model_path = "/home/shared_data/zdukb/POPDx/code/save/data_5_10_pretrain"
    # biotokenizer = AutoTokenizer.from_pretrained(model_path)
    # bert = AutoModel.from_pretrained(model_path)
    
    bert.to(device)
    bert.eval()
    
    # with open('../data/mc_icd10_labels.txt','r') as f:
    #     labels = f.readlines()
    # labels = [x.strip() for x in labels] 
    labels = pd.read_csv("/home/shared_data/zdukb/POPDx/data/icd10_meaning.csv")["ICD10"].tolist()
    coding19 = pd.read_csv('../data/19.csv')
    # ipdb.set_trace()
    df_labels = pd.merge(pd.DataFrame(labels, columns=['coding']), coding19, on='coding', how='left')
    batch_all = df_to_batch(df_labels, batch_size)
    devices_k = []
    features_all = []
    pooled_all = []
    for batch in batch_all:
        print('Batch #', batch)
        # ipdb.set_trace()
        #k, (out_hidden_states, pooled_states) = biobert_embed(batch_all[batch], bert, biotokenizer, device)
        # 'last_hidden_state'
        # 'pooler_output'
        
        k, hidden_tuple,attention_mask = biobert_embed(batch_all[batch], bert, biotokenizer, device)
        out_hidden_states = hidden_tuple['last_hidden_state']
        pooled_states = hidden_tuple['pooler_output']
        # last_hidden_states = out_hidden_states.cpu()
        
        last_hidden_states = out_hidden_states
        features = torch.sum(last_hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)
        features = torch.div(features,padd_len)
        # ipdb.set_trace()
        devices_k.append(k) 
        features_all.append(features.cpu())
        pooled_all.append(pooled_states.cpu())
    devices_lst = np.concatenate(devices_k, axis=0)
    features = np.concatenate(features_all, axis=0)
    pooled = np.concatenate(pooled_all, axis=0)
    biboert_embeddings_dict = dict(zip(devices_lst,features))
    np.savez('/home/shared_data/zdukb/POPDx/data/data_icd_text/biobert_embeddings.npz', **biboert_embeddings_dict)
    np.save('/home/shared_data/zdukb/POPDx/data/data_icd_text/biobert_embeddings.npy', features)
    
    ipdb.set_trace()
    assert labels == list(biboert_embeddings_dict.keys())
    return features, biboert_embeddings_dict

def run_bert_text(batch_size=128, use_pretrain = True):
    # if use_pretrain:
    #     try:
    #         biboert_embeddings_dict = dict(np.load(os.path.join('./embeddings/biobert_embeddings_dict.npz')))
    #         features = np.load(os.path.join('./embeddings/biobert_embeddings.npy'))
    #         return features, biboert_embeddings_dict
    #     except:
    #         print('No pretrained BERT embeddings found.')

    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"  
    print(device)
    # biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    # bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert.to(device)
    bert.eval()
    
    df = pd.read_csv("/home/shared_data/zdukb/POPDx/coding/text_new2.csv").set_index("eid")
    batch_all = df_to_batch(df, batch_size)
    devices_k = []
    features_all = []
    pooled_all = []
    progress_bar = tqdm(range(len(df)))
    for batch in batch_all:
        print('Batch #', batch)
        # ipdb.set_trace()
        #k, (out_hidden_states, pooled_states) = biobert_embed(batch_all[batch], bert, biotokenizer, device)
        # 'last_hidden_state'
        # 'pooler_output'
        progress_bar.update(1)
        hidden_tuple,attention_mask = biobert_embed_text(batch_all[batch], bert, biotokenizer, device)
        out_hidden_states = hidden_tuple['last_hidden_state']
        # pooled_states = hidden_tuple['pooler_output']
        last_hidden_states = out_hidden_states
        features = torch.sum(last_hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)
        features = torch.div(features,padd_len)
        features_all.append(features.cpu())
        # pooled_all.append(pooled_states.cpu())
    features = np.concatenate(features_all, axis=0)
    # pooled = np.concatenate(pooled_all, axis=0)
    ipdb.set_trace()
    np.save("/home/shared_data/zdukb/POPDx/data/data_5_25_icd/text_embedding_531.npy", features)
    # ipdb.set_trace()

    return features

def run_bert_text_short(batch_size=64, use_pretrain = True, file=None, max_len=512):
    # path = os.path.join('/home/shared_data/zdukb/POPDx/coding/text', file+'.csv')
    df = pd.read_csv("/home/shared_data/zdukb/POPDx/coding/text_field12_sep.csv")
    # df = pd.read_csv(path)
    text = df["0"]
    text_list = [text[i] for i in range(len(text))]

    device = torch.device("cuda:0" if True else "cpu")
    features_all = []
    att_all = []
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert.to(device)
    bert.eval()
    text_data = Dataset_embed(text_list)
    dataloader = DataLoader(text_data, batch_size=batch_size, shuffle=False)
    progress_bar = tqdm(range(len(text_data)))
    
    for batch in dataloader:
        encoded_bh= biotokenizer(batch, padding="max_length", max_length = max_len, truncation=True, return_tensors="pt",  add_special_tokens = False)
        input_ids, attention_mask =  Variable(encoded_bh["input_ids"].to(device)), Variable(encoded_bh["attention_mask"].to(device))
        progress_bar.update(batch_size)
        with torch.no_grad():
            hidden_states = bert(input_ids, attention_mask=attention_mask)['last_hidden_state']

        # ipdb.set_trace()
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        features = torch.div(text_embed,padd_len)
        features_all.append(features.cpu().numpy())


    features = np.concatenate(features_all, axis=0)

    np.save("/home/shared_data/zdukb/POPDx/data/data_icd_text/text_embedding_field12_sep.npy", features)

    return features


def run_bert_phecode(batch_size=256, use_pretrain = True):
    if use_pretrain:
        try:
            biboert_embeddings_dict = dict(np.load(os.path.join('./embeddings/biobert_embeddings_dict_phecode.npz')))
            features = np.load(os.path.join('./embeddings/biobert_embeddings_phecode.npy'))
            return features, biboert_embeddings_dict
        except:
            print('No pretrained BERT embeddings found.')
    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"  
    print(device)
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    bert.to(device)
    bert.eval()
    
    with open('../data/phecode_labels.txt','r') as f:
        labels = f.readlines()
    labels = ["{:0.2f}".format(float(x.strip())) for x in labels] #"{:0.2f}".format(float(x.strip()))
    coding20 = pd.read_csv('../data/20.csv').iloc[0:1865,:]
    coding20['phecode'] = coding20['phecode'].astype('float')
    phecode_labels =pd.DataFrame(labels, columns=['phecode'])
    phecode_labels['phecode'] = phecode_labels['phecode'].astype('float')
    # ipdb.set_trace()
    df_labels = pd.merge(phecode_labels, coding20, on='phecode')
    batch_all = df_to_batch(df_labels, batch_size)
    devices_k = []
    features_all = []
    pooled_all = []
    for batch in batch_all:
        print('Batch #', batch)
        #k, (out_hidden_states, pooled_states) = biobert_embed(batch_all[batch], bert, biotokenizer, device)
        # 'last_hidden_state'
        # 'pooler_output'
        
        k, hidden_tuple,attention_mask = biobert_embed_phecode(batch_all[batch], bert, biotokenizer, device)
        out_hidden_states = hidden_tuple['last_hidden_state']
        pooled_states = hidden_tuple['pooler_output']
        last_hidden_states = out_hidden_states
        features = last_hidden_states[:, 0, :]
        # features = torch.sum(last_hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)
        # features = torch.div(features,padd_len)
        
        devices_k.append(k) 
        features_all.append(features.cpu())
        pooled_all.append(pooled_states.cpu())
    devices_lst = np.concatenate(devices_k, axis=0)
    features = np.concatenate(features_all, axis=0)

    pooled = np.concatenate(pooled_all, axis=0)
    biboert_embeddings_dict = dict(zip(devices_lst,features))
    
    np.savez(os.path.join('./embeddings/biobert_embeddings_dict_phecode.npz'), **biboert_embeddings_dict)
    np.save(os.path.join('./embeddings/biobert_embeddings_phecode.npy'), features)
    assert labels == list(biboert_embeddings_dict.keys())
    return features, biboert_embeddings_dict

def run_bert_long(batch_size=64, use_pretrain = True, file=None, max_len=512):
    df = pd.read_csv("/home/shared_data/zdukb/POPDx/coding/text_field12.csv")
    device = torch.device("cuda:0" if True else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    # num=0
    text = df["0"]
    # ipdb.set_trace()
    text_list = [text.iloc[i] for i in range(len(text))]
    text_data = Dataset_embed(text_list)
    dataloader = DataLoader(text_data, batch_size=batch_size, shuffle=False)
    progress_bar = tqdm(range(len(text_data)))
    bert = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    bert.to(device)
    bert.eval()
    max_len = 640
    features_all = []
    for i,batch in enumerate(dataloader):
        print('Batch #', i)
        encoded_bh= tokenizer(batch, padding="max_length", max_length = max_len, truncation=True, return_tensors="pt",  add_special_tokens = False)
        input_ids, attention_mask =  Variable(encoded_bh["input_ids"].to(device)), Variable(encoded_bh["attention_mask"].to(device))
        progress_bar.update(batch_size)
        with torch.no_grad():
            hidden_states = bert(input_ids, attention_mask=attention_mask)['last_hidden_state']
    
        text_embed= torch.sum(hidden_states, dim=-2)
        padd_len = (torch.sum(attention_mask,dim=-1)).unsqueeze(-1)    
        features = torch.div(text_embed,padd_len)
        features_all.append(features.cpu().numpy())
        # ipdb.set_trace()
    features = np.concatenate(features_all, axis=0)
    np.save(os.path.join('../data/data_icd_text/text_embedding_0607.npy'), features)

def debug(batch_size=64, use_pretrain = True, file=None, max_len=640):
    df = pd.read_csv("/home/shared_data/zdukb/POPDx/coding/text_new2.csv")
    device = torch.device("cuda:0" if True else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    # num=0
    text = df["0"]
    # ipdb.set_trace()
    text_list = [text.iloc[i] for i in range(len(text))]
    text_data = Dataset_embed(text_list)
    dataloader = DataLoader(text_data, batch_size=batch_size, shuffle=False)
    progress_bar = tqdm(range(len(text_data)))
    truncated_tokens = []
    for i, batch in enumerate(dataloader):
        print('Batch #', i)
        progress_bar.update(batch_size)
        encoded_bh= tokenizer(batch, padding="max_length", max_length = max_len, truncation=True, return_tensors="pt",  add_special_tokens = False)
        rows = encoded_bh["input_ids"].shape[0]
        for j in range(rows):
            if encoded_bh["input_ids"][j,512]!= torch.tensor(1):
                # ipdb.set_trace()
                tokens_ = tokenizer.convert_ids_to_tokens(encoded_bh["input_ids"][j,512:max_len])
                last_non_pad_index = len(tokens_) - 1
                while last_non_pad_index >= 0 and tokens_[last_non_pad_index] == '<pad>':
                    last_non_pad_index -= 1
                tokens_ = tokens_[:last_non_pad_index+1]
                my_string = ' '.join(tokens_)
                truncated_tokens.append(my_string)
                print(my_string)
    with open('../coding/truncated_tokens.txt', 'w') as f:
        for item in truncated_tokens:
            f.write(item + '\n')



if __name__ == "__main__":
    #(l2i, i2l, X_emb) = phecode_ontoemb(dim=500, ICD_network_file = '../data/20.csv', save_dir = './onto_embeddings/', use_pretrain = True)
    #biboert_embeddings, biboert_embeddings_dict = run_bert_phecode(use_pretrain = False)
    #ontology_emb(dim=250, ICD_network_file = '../data/19.csv', save_dir = './icd_embeddings/', use_pretrain = True)
    #ontology_emb(dim=500, ICD_network_file = '../data/19.csv', save_dir = './icd_embeddings/', use_pretrain = False)
    # biboert_embeddings, biboert_embeddings_dict = run_bert(use_pretrain = False)
    
    #features = run_bert_text_short(batch_size=256, use_pretrain = True)

    # fields = ["20001","20002","20003","20004","22601","40006","40011","40013","41200","41210","41245","41246"]
    # record_coding = pd.read_excel('/home/shared_data/zdukb/POPDx/coding/record_coding_new.xlsx',sheet_name=["Sheet2"])["Sheet2"]
    # record_dict = dict(zip(record_coding['field'], record_coding['max_len']))
    # for field in fields:
    #     file = 'text_' + field
    #     max_len = record_dict[int(field)]
    #     features = run_bert_text_short(batch_size=256, use_pretrain=True, file=file, max_len=max_len)
    #debug(batch_size=64, use_pretrain = True)
    # run_bert_text_short(batch_size=128, use_pretrain = True, file=None, max_len=512)
    phecode_ontoemb(dim=500)