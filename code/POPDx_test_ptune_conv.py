import torch
import torch.nn as nn
import time
from transformers import AutoModel, BertPreTrainedModel, BertConfig, AutoTokenizer
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import BertPrefixForMLclassification_conv,BertPrefixForMLclassification_conv_normal3,BertPrefixForMLclassification_convnn,BertPrefixForMLclassification_deeplift, BertPrefixForMLcls_conv_onlytext, BertPrefixForMLcls_convnn_onlytext
from data import *
from tools import ModelSaving
from logger import *
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve,confusion_matrix,f1_score


def args_argument():
    parser = argparse.ArgumentParser(description='''
    The script to test POPDx. 
    Please specify the path to the test datasets in the python script.
    ''')
    parser.add_argument('-d', '--save_dir', required=True, help='The path to POPDx model e.g. "./save/POPDx_train"')
    # parser.add_argument('-bs', '--batch_size', required=True, default=138, help='The batch size"')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pre_seq_len", type=int, default=50)
    parser.add_argument('-s', '--hidden_size', type=int, default=150,
                        help='Default hidden size is 150. Consistent with training.')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Default batch size is 513.')
    parser.add_argument('-max_len', '--max_length', type=int, default=512, help='Default max length is 64')
    parser.add_argument('-bh', '--bert_hidden_size', type=int, default=768, help='Default max length is 64')
    parser.add_argument('--use_gpu', default=True, help='Default setup is to not use GPU for test.')
    parser.add_argument('--debug', type=int, default=None)

    parser.add_argument('--classifi',type=str, default="icd")

    parser.add_argument('--self_loop',type = int, default = 2)
    parser.add_argument('--model_name',type = str,required=True)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--net",type=str,default="conv")
    parser.add_argument('--try1',type=int,default=3)
    parser.add_argument('--try2',type=int,default=0)
    parser.add_argument('--lower_weight', type=float, default=0.9)
    parser.add_argument('--upper_weight', type=float, default=0.7)
    args = parser.parse_args()
    return args

def generate_cv_icd(args):
    if args.try1!=3:
        adj = np.load('./embeddings/adj.npy',allow_pickle=True)
    elif args.try1==3:
        #给树状图不同指向的边赋予不同的权重
        # adj下三角赋大权重，因为有A001一定有A00，上三角赋小权重有A00可能会有A001
        adj = np.load('./rearrange_embeddings/icd_adj_arrange.npy',allow_pickle=True)
        ones_lower = np.tril(np.ones_like(adj))

        # 生成一个全1矩阵，用于上三角部分
        ones_upper = np.triu(np.ones_like(adj))

        # 将下三角部分与小权重相乘，上三角部分与大权重相乘
        adj = adj * (ones_lower * args.lower_weight + ones_upper * args.upper_weight)



    I = args.self_loop*np.eye(adj.shape[0])
        
    k = args.alpha
    
    if args.try2 ==0:
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A_1 = invsqrt_degree_matrix@adj@invsqrt_degree_matrix

        adj2 = adj*adj
        invsqrt_degree_matrix2 = np.diag(1/np.sqrt(np.sum(adj2, axis=1)))
        A_2 = invsqrt_degree_matrix2@adj2@invsqrt_degree_matrix2

        adj3 = adj*adj*adj
        invsqrt_degree_matrix3 = np.diag(1/np.sqrt(np.sum(adj3, axis=1)))
        A_3 = invsqrt_degree_matrix3@adj3@invsqrt_degree_matrix3
        conv = I+ k*A_1+ k*k*A_2 + k*k*k*A_3    
    # elif args.try2 ==1:
    #     conv = I+ k*A_1+ k*A_2 + k*A_3 
    # elif args.try2 ==2:   
    #     conv = I+ k*A_1     

    elif args.try2 ==3:

        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj+I, axis=1)))
        A = invsqrt_degree_matrix@adj@invsqrt_degree_matrix
        conv = A@A@A

    elif args.try2 ==4:
        conv = I+ k*A_1+ k*k*A_2 + k*k*k*A_3 
    elif args.try2==5:
        
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A_1 = invsqrt_degree_matrix@adj@invsqrt_degree_matrix

        adj2 = adj@adj
        invsqrt_degree_matrix2 = np.diag(1/np.sqrt(np.sum(adj2, axis=1)))
        A_2 = invsqrt_degree_matrix2@adj2@invsqrt_degree_matrix2

        adj3 = adj@adj@adj
        invsqrt_degree_matrix3 = np.diag(1/np.sqrt(np.sum(adj3, axis=1)))
        A_3 = invsqrt_degree_matrix3@adj3@invsqrt_degree_matrix3

        conv = I+ k*A_1+ k*k*A_2 + k*k*k*A_3 
    
    elif args.try2 ==6 :

        invsqrt_degree_matrix = np.diag(1/np.sum(adj+I, axis=1))
        A = invsqrt_degree_matrix@adj
        conv = A@A@A

    elif args.try2 ==7:
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A = invsqrt_degree_matrix@adj@invsqrt_degree_matrix
        
        conv = I+ k*A+ k*k*A@A + k*k*k*A@A@A     

    with open('../data/mc_icd10_labels.txt','r') as f:
        labels = f.readlines()
    labels = [s.strip() for s in labels]
    with open("./embeddings/embedding_node_to_idx_dict.json", 'r') as inpt:
        l2i = json.load(inpt)
    labels_idx = [l2i[l] for l in labels]
    conv= conv[labels_idx,:][:,labels_idx]
    print("done generating convolution matrix")

    return conv

def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    if isinstance(n, int):
        return n
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        return n

def generate_cv_phe(args):
    
    if args.try1!=3:
        adj = np.load('./onto_embeddings/adj_onto.npy',allow_pickle=True)
    elif args.try1==3:
        #给树状图不同指向的边赋予不同的权重
        # adj下三角赋大权重，因为有A001一定有A00，上三角赋小权重有A00可能会有A001
        adj = np.load('./rearrange_embeddings/phe_adj_arrange.npy',allow_pickle=True)
        ones_lower = np.tril(np.ones_like(adj))

        # 生成一个全1矩阵，用于上三角部分
        ones_upper = np.triu(np.ones_like(adj))

        # 将下三角部分与小权重相乘，上三角部分与大权重相乘
        adj = adj * (ones_lower * args.lower_weight + ones_upper * args.upper_weight)
    I = args.self_loop*np.eye(adj.shape[0])
    
    k = args.alpha


    if args.try2 ==0:
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A_1 = invsqrt_degree_matrix@adj@invsqrt_degree_matrix

        adj2 = adj*adj
        invsqrt_degree_matrix2 = np.diag(1/np.sqrt(np.sum(adj2, axis=1)))
        A_2 = invsqrt_degree_matrix2@adj2@invsqrt_degree_matrix2

        adj3 = adj*adj*adj
        invsqrt_degree_matrix3 = np.diag(1/np.sqrt(np.sum(adj3, axis=1)))
        A_3 = invsqrt_degree_matrix3@adj3@invsqrt_degree_matrix3
        conv = I+ k*A_1+ k*k*A_2 + k*k*k*A_3    
    # elif args.try2 ==1:
    #     conv = I+ k*A_1+ k*A_2 + k*A_3        
    # elif args.try2 ==2:   
    #     conv = I+ k*A_1
    elif args.try2 ==3:
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj+I, axis=1)))
        A = invsqrt_degree_matrix@adj@invsqrt_degree_matrix
        conv = A@A@A
    elif args.try2 ==4:
        invsqrt_degree_matrix = np.diag(1/np.sum(adj+I, axis=1))
        A = invsqrt_degree_matrix@adj
        conv = A@A@A

    elif args.try2==5:
        
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A_1 = invsqrt_degree_matrix@adj@invsqrt_degree_matrix

        adj2 = adj@adj
        invsqrt_degree_matrix2 = np.diag(1/np.sqrt(np.sum(adj2, axis=1)))
        A_2 = invsqrt_degree_matrix2@adj2@invsqrt_degree_matrix2

        adj3 = adj@adj@adj
        invsqrt_degree_matrix3 = np.diag(1/np.sqrt(np.sum(adj3, axis=1)))
        A_3 = invsqrt_degree_matrix3@adj3@invsqrt_degree_matrix3

        conv = I+ k*A_1+ k*k*A_2 + k*k*k*A_3 
    
    elif args.try2 ==6 :

        invsqrt_degree_matrix = np.diag(1/np.sum(adj+I, axis=1))
        A = invsqrt_degree_matrix@adj
        conv = A@A@A

    elif args.try2 ==7:
        invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
        A = invsqrt_degree_matrix@adj@invsqrt_degree_matrix
        
        conv = I+ k*A+ k*k*A@A + k*k*k*(A@A@A + A@A@A@A)
    phecode_labels = pd.read_csv("/home/shared_data/zdukb/COVID-19/process_5_4/phecode_csv.csv")["PHECODE"].tolist()
    
    if args.try1!=3:
        with open("./onto_embeddings/embedding_node_to_idx_dict.json", 'r') as inpt:
            l2i = json.load(inpt)
    elif args.try1==3:
        with open("./rearrange_embeddings/phe_embedding_node_to_idx_dict.json", 'r') as inpt:
            l2i = json.load(inpt)
    

    labels_idx = [l2i[str(delete_extra_zero(l))] for l in phecode_labels]
    conv= conv[labels_idx,:][:,labels_idx]
    print("done generating convolution matrix")
    # ipdb.set_trace()
    return conv

    


def load_data(args):
    if args.classifi == 'icd':
        data_folder = '/home/shared_data/zdukb/POPDx/data_new/icd' #data
        test_feature_file = "test_feature_16.npy"
        test_label_file = 'test_icd_labels.npy'
        val_feature_file = "val_feature_16.npy"
        val_label_file = 'val_icd_labels.npy'
        train_label_file = "train_icd_labels.npy"
        label_emb = generate_cv_icd(args)
    else:
        data_folder = '/home/shared_data/zdukb/POPDx/data_new/phecode' #data
        
        test_feature_file = 'test_feature_16.npy'
        test_label_file = 'test_phe_labels.npy'
        val_feature_file = "val_feature_16.npy"
        val_label_file = 'val_phe_labels.npy'
        train_label_file = "train_phe_labels.npy"
        label_emb = generate_cv_phe(args)

    test_feature = np.load(os.path.join(data_folder, test_feature_file), allow_pickle=True)
    val_feature = np.load(os.path.join(data_folder, val_feature_file), allow_pickle=True)
    test_label = np.load(os.path.join(data_folder, test_label_file), allow_pickle=True) 
    val_label = np.load(os.path.join(data_folder, val_label_file), allow_pickle=True) 
    train_label = np.load(os.path.join(data_folder, train_label_file), allow_pickle=True) 

    print(test_feature.shape)


    return train_label,val_feature,val_label, test_feature,test_label,label_emb



def test(train_label, val_feature, test_feature, val_label, test_label, label_emb,
          model_checkpoint_loc=None, output_dir=None, use_cuda=False, hidden_size=150, batch_size=512):

    device = torch.device("cuda:0" if use_cuda else "cpu")
    label_emb = torch.tensor(label_emb, dtype=torch.float, device=device)


    if args.classifi == "icd":
            val_file = "/home/shared_data/zdukb/POPDx/data_new/icd/val_text_raw12.txt"
            test_file = "/home/shared_data/zdukb/POPDx/data_new/icd/test_text_raw12.txt"
    else:
            val_file = "/home/shared_data/zdukb/POPDx/data_new/phecode/val_text_raw12_phe.txt"
            test_file = "/home/shared_data/zdukb/POPDx/data_new/phecode/test_text_raw12_phe.txt"

    with open(val_file, "r", encoding='utf-8') as file:
        val_text = [line.strip() for line in file.readlines()]

    with open(test_file, "r", encoding='utf-8') as file:
        test_text = [line.strip() for line in file.readlines()]

    if args.debug == 1:
        with open(val_file, "r", encoding='utf-8') as file:
            val_text = [line.strip() for line in file.readlines()[:2000 * args.batch_size]]

        with open(test_file, "r", encoding='utf-8') as file:
            test_text = [line.strip() for line in file.readlines()[:2000 * args.batch_size]]

        val_feature = val_feature[:2000 * args.batch_size, :]
        test_feature = test_feature[:2000 * args.batch_size, :]
        
    # Load Model
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    config = BertConfig.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    if args.net =="conv":
        net = BertPrefixForMLclassification_conv(config,test_feature.shape[1],hidden_size,label_emb,args.pre_seq_len)
    if args.net == "conv_normal3":
        net = BertPrefixForMLclassification_conv_normal3(config, test_feature.shape[1], hidden_size, label_emb,
                                                 args.pre_seq_len)
    elif args.net == "conv_onlytext":
        net = BertPrefixForMLcls_conv_onlytext(config, hidden_size, label_emb, args.pre_seq_len)
    elif args.net == "convnn_onlytext":
        net = BertPrefixForMLcls_convnn_onlytext(config, hidden_size, label_emb, args.pre_seq_len)
    elif args.net == "nn":
        net = BertPrefixForMLclassification_convnn(config,test_feature.shape[1],hidden_size,label_emb,args.pre_seq_len)
    elif args.net == "conv_comb":
        if args.classifi == "phe":
            phe_coef = pd.read_excel("../data/phe_coef.xlsx",sheet_name=["Sheet1"])["Sheet1"]
            phe_coef = torch.tensor(phe_coef.coef.to_numpy(), device=device)
            eye = torch.diag_embed(args.self_loop*torch.ones_like(phe_coef))
            phe_coef = phe_coef.unsqueeze(0)
            comb_matrix = phe_coef*eye + (1-phe_coef)*label_emb
            comb_matrix = comb_matrix.to(torch.float32)
            net = BertPrefixForMLclassification_conv(config,test_feature.shape[1],hidden_size,comb_matrix,args.pre_seq_len)
        elif args.classifi =="icd":
            icd_coef = pd.read_excel("../data/icd_coef.xlsx",sheet_name=["Sheet1"])["Sheet1"]
            icd_coef = torch.tensor(icd_coef.coef.to_numpy(), device=device)
            eye = torch.diag_embed(args.self_loop*torch.ones_like(icd_coef))
            icd_coef = icd_coef.unsqueeze(0)
            comb_matrix = icd_coef*eye + (1-icd_coef)*label_emb
            comb_matrix = comb_matrix.to(torch.float32)
            net = BertPrefixForMLclassification_convnn(config,test_feature.shape[1],hidden_size,comb_matrix,args.pre_seq_len)
    elif args.net == "conv_comb1":
        if args.classifi == "phe":
            phe_coef = pd.read_excel("../data/phe_coef1.xlsx",sheet_name=["Sheet1"])["Sheet1"]#加conv的部分更少
            phe_coef = torch.tensor(phe_coef.coef.to_numpy(), device=device)
            eye = torch.diag_embed(args.self_loop*torch.ones_like(phe_coef))
            phe_coef = phe_coef.unsqueeze(0)
            comb_matrix = phe_coef*eye + (1-phe_coef)*label_emb
            comb_matrix = comb_matrix.to(torch.float32)
            net = BertPrefixForMLclassification_conv(config,test_feature.shape[1],hidden_size,comb_matrix,args.pre_seq_len)
        elif args.classifi =="icd":
            icd_coef = pd.read_excel("../data/icd_coef.xlsx",sheet_name=["Sheet1"])["Sheet1"]
            icd_coef = torch.tensor(icd_coef.coef.to_numpy(), device=device)
            eye = torch.diag_embed(args.self_loop*torch.ones_like(icd_coef))
            icd_coef = icd_coef.unsqueeze(0)
            comb_matrix = icd_coef*eye + (1-icd_coef)*label_emb
            comb_matrix = comb_matrix.to(torch.float32)
            net = BertPrefixForMLclassification_convnn(config,test_feature.shape[1],hidden_size,comb_matrix,args.pre_seq_len)                         
    elif args.net == "lasso":
        net = BertPrefixForMLclassification_deeplift(config,args.pre_seq_len)

    if use_cuda:
        net.cuda()
    if args.net == "lasso":
        bert_state_dict = {}
        checkpoint = torch.load(model_checkpoint_loc, map_location=device)
        for key, value in checkpoint['model_state_dict'].items():
            if key not in ['linears.0.weight', 'linears.0.bias', 'linears.1.weight', 'linears.1.bias']:
                bert_state_dict[key] = value
        net.load_state_dict(bert_state_dict)
    else:
        checkpoint = torch.load(model_checkpoint_loc, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    print('Done loading model.')

    train_icd_num = pd.DataFrame(train_label.sum(axis=0))
    val_icd_num = pd.DataFrame(val_label.sum(axis=0))
    test_icd_num = pd.DataFrame(test_label.sum(axis=0))
    # Load Data
    def collate_fn(batch):
        # 将句子列表和NumPy数据分离
        text = [item["text"] for item in batch]
        x_data = torch.stack([item["x_data"] for item in batch])
        y_data = torch.stack([item["y_data"] for item in batch])

        # 使用BERT分词器对句子进行分词和编码
        encoded_sentences = biotokenizer(text, padding="max_length", max_length=args.max_length - args.pre_seq_len,
                                         truncation=True, return_tensors="pt", add_special_tokens=True)
        return {
            "input_ids": encoded_sentences["input_ids"],
            "attention_mask": encoded_sentences["attention_mask"],
            "x_data": x_data,
            "y_data": y_data
        }

    net.eval()

    val_dataset = Dataset_pipline(val_feature, val_label, val_text)
    test_dataset = Dataset_pipline(test_feature, test_label, test_text)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    test_outputs = []
    test_truth = []
    if args.net == "lasso":
        train_feature = np.load("/home/shared_data/zdukb/POPDx/data_new/phecode/train_feature_16.npy", allow_pickle=True)
        with open("/home/shared_data/zdukb/POPDx/data_new/phecode/train_text_raw12_phe.txt", "r", encoding='utf-8') as file:
            train_text = [line.strip() for line in file.readlines()]  
        train_dataset = Dataset_pipline(train_feature, train_label, train_text)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        train_outputs = []
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            train_inputs, train_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
            train_output = net(input_ids, attention_mask)
            train_outputs.append(train_output.detach().cpu())
        train_outputs = torch.cat(train_outputs, dim=0)
        np.save("/home/shared_data/zdukb/POPDx/save_new/phe_724_analysis/train_text_embedding.npy",train_outputs.numpy())
        ipdb.set_trace()
    elif (args.net == "conv_onlytext") or (args.net == "convnn_onlytext"):
        for batch_idx, batch in tqdm(enumerate(test_loader)):
            test_inputs, test_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
            test_output = net(input_ids, attention_mask)
            test_outputs.append(test_output.detach().cpu())
            test_truth.append(test_labels.detach().cpu())
        test_outputs = torch.cat(test_outputs, dim=0)
        test_truth = torch.cat(test_truth, dim=0)
    else:
        for batch_idx, batch in tqdm(enumerate(test_loader)):
            test_inputs, test_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
            test_output = net(test_inputs, input_ids, attention_mask)
            test_outputs.append(test_output.detach().cpu())
            test_truth.append(test_labels.detach().cpu())
        test_outputs = torch.cat(test_outputs, dim=0)
        test_truth = torch.cat(test_truth, dim=0)
    print('Generating the test statistics!')

    val_outputs = []
    val_truth = []
    if (args.net == "conv_onlytext") or (args.net == "convnn_onlytext"):
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            val_inputs, val_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(
                batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(
                batch["attention_mask"].to(device))
            val_output = net(input_ids, attention_mask)
            val_outputs.append(val_output.detach().cpu())
            val_truth.append(val_labels.detach().cpu())
        val_outputs = torch.cat(val_outputs, dim=0)
        val_truth = torch.cat(val_truth, dim=0)
    else:
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            val_inputs, val_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
            val_output = net(val_inputs, input_ids, attention_mask)
            val_outputs.append(val_output.detach().cpu())
            val_truth.append(val_labels.detach().cpu())
        val_outputs = torch.cat(val_outputs, dim=0)
        val_truth = torch.cat(val_truth, dim=0)
    print('Generating the valid statistics!')

    # Find Threshold
    col_sum = np.sum(test_label, axis=0)
    nonzero_col_idx = np.where(col_sum != 0)[0]
    threshold = {i: None for i in nonzero_col_idx}  # 将那些测试集有正样本的label取出来先初始化字典
    threshold = find_threshold(val_outputs, val_label, threshold)
    print("Done finding threshold!")

    # Test Metrics
    sampling_test(test_outputs, test_truth, test_label, output_dir, threshold,train_icd_num,val_icd_num,test_icd_num, neg_to_pos_ratio=10)

    val_outputs_nonzero = val_outputs[:, nonzero_col_idx]
    val_label_nonzero = val_label[:, nonzero_col_idx]
    threshold_micro = find_threshold_micro(val_outputs_nonzero, val_label_nonzero)
    print("Done finding micro threshold!")

    train_col_sum = np.sum(train_label, axis=0)
    top_50_cols = np.argsort(train_col_sum)[-50:]
    threshold_macro = {i: None for i in top_50_cols}
    threshold_macro = find_threshold_macro(val_outputs, val_label, threshold_macro)
    print("Done finding macro threshold!")


    print("Macro and micro!")
    MMAF(test_outputs, test_truth, threshold, threshold_micro, threshold_macro, output_dir)

    print('done')


def MMAF(test_outputs, test_truth, threshold, threshold_micro, threshold_macro, output_dir):
    idx = list(threshold.keys())
    test_outputs_nonzero = test_outputs[:, idx]
    test_truth_nonzero = test_truth[:, idx]
    threshold_values = list(threshold.values())

    idx_top50 = list(threshold_macro.keys())
    test_outputs_top50 = test_outputs[:, idx_top50]
    test_truth_top50 = test_truth[:, idx_top50]
    threshold_values_top50 = list(threshold_macro.values())

    predicted_test = (torch.sigmoid(test_outputs_nonzero).data > torch.tensor(threshold_values)).type(torch.float)
    predicted_test_micro = (torch.sigmoid(test_outputs_nonzero).data > threshold_micro).type(torch.float)
    predicted_test_macro = (torch.sigmoid(test_outputs_top50).data > torch.tensor(threshold_values_top50)).type(
        torch.float)
    micro_auc = roc_auc_score(test_truth_nonzero, test_outputs_nonzero, average='micro')
    macro_auc = roc_auc_score(test_truth_nonzero, test_outputs_nonzero, average='macro')
    micro_f1 = f1_score(test_truth_nonzero, predicted_test_micro, average='micro')
    macro_f1 = f1_score(test_truth_nonzero, predicted_test, average='macro')
    macro_f1_top50 = f1_score(test_truth_top50, predicted_test_macro, average='macro')
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        print('Micro AUC: ', micro_auc, file=f)
        print('Macro AUC: ', macro_auc, file=f)
        print('Micro F1: ', micro_f1, file=f)
        print('Macro F1: ', macro_f1, file=f)
        print('Macro F1@50: ', macro_f1_top50, file=f)


def sampling_test(test_outputs, test_truth, test_label, output_dir, threshold,train_icd_num,val_icd_num,test_icd_num, neg_to_pos_ratio=10):
    print('Setting the negative to positive ratio.')
    ipdb.set_trace()
    # precision_ = {}
    # recall_ = {}
    AUPRC = {}
    # fpr = {}
    # tpr = {}
    roc_auc = {}
    f1 = {}

    for i in tqdm(range(test_label.shape[1])):
        results = {}
        y_test_subset = test_label[:, i]
        idx_icd10 = [i for i, j in enumerate(y_test_subset) if j == 1]
        idex_neg = [i for i, j in enumerate(y_test_subset) if j != 1]
        if len(idx_icd10) > 0:  # NOTE:测试集0样本的label不会被计算

            test_outputs_i = []
            test_truth_i = []
            results[i] = []
            if len(idex_neg) / len(idx_icd10) >= 10:
                # if len(idx_icd10)<1000:
                # print(i,'few minority', len(idx_icd10))
                for j in range(50):  # 共选了800个正样本
                    selected_pos = random.choices(idx_icd10, k=16)  # 选择了患该病的subjects里的16个人
                    selected_neg = random.sample(idex_neg, 16 * neg_to_pos_ratio)  # 选择了160个不患该病的人
                    selected_ = selected_pos + selected_neg
                    test_outputs_i.append(test_outputs[selected_])
                    test_truth_i.append(test_truth[selected_])
            else:
                # print(i,'majority majority')
                for j in range(1500):  # 2400个正样本
                    selected_pos = random.sample(idx_icd10, 16)
                    selected_neg = random.sample(idex_neg, 16 * neg_to_pos_ratio)
                    selected_ = selected_pos + selected_neg
                    test_outputs_i.append(test_outputs[selected_])
                    test_truth_i.append(test_truth[selected_])

            test_outputs_ = torch.cat(test_outputs_i, dim=0)
            test_truth_ = torch.cat(test_truth_i, dim=0)
            probs_all_ = torch.sigmoid(test_outputs_).data
            predicted_test_ = (torch.sigmoid(test_outputs_).data > threshold[i]).type(torch.float)
            results[i] = [test_outputs_[:, i], test_truth_, predicted_test_[:, i], probs_all_[:, i]]
            #

        else:
            continue

        # precision_[i], recall_[i], _ = precision_recall_curve(results[i][1][:,i].numpy(), results[i][3].numpy())
        # AUPRC[i] = auc(recall_[i], precision_[i])
        precision, recall, _ = precision_recall_curve(results[i][1][:, i].numpy(), results[i][3].numpy())
        AUPRC[i] = auc(recall, precision)
        # fpr[i], tpr[i], _ = roc_curve(results[i][1][:,i].numpy(), results[i][3].numpy())
        roc_auc[i] = roc_auc_score(results[i][1][:, i].numpy(), results[i][3].numpy(), average='macro')
        f1[i] = f1_score(results[i][1][:, i].numpy(), results[i][2].numpy(), average='binary')
    
    icd_models = [args.model_name]
    icd_columns = ["train_icd_num","val_icd_num","test_icd_num"]


    icd_model_auc = pd.DataFrame.from_dict(roc_auc,orient="index")
    icd_model_f1 = pd.DataFrame.from_dict(f1,orient="index")

    icd_concat = pd.concat([train_icd_num, val_icd_num, test_icd_num,
                        icd_model_auc, icd_model_f1],axis=1)
    icd_concat.to_csv(os.path.join(output_dir, args.model_name+".csv"))

    for model in icd_models:
        icd_columns.append(model+'_auc')
        icd_columns.append(model + '_f1')
    icd_concat.columns = icd_columns

    icd_results = pd.DataFrame(0, index=range(6), columns=icd_columns[3:])
    for model in icd_models:
        ROC_AUC = []
        F1 = []
        ROC_AUC.append(icd_concat[icd_concat['train_icd_num'] == 0][model+'_auc'].mean())
        ROC_AUC.append(icd_concat[(icd_concat['train_icd_num'] > 0) & (icd_concat['train_icd_num'] <= 10)][model+'_auc'].mean())
        ROC_AUC.append(icd_concat[(icd_concat['train_icd_num'] >10) & (icd_concat['train_icd_num'] <= 100)][model + '_auc'].mean())
        ROC_AUC.append(icd_concat[(icd_concat['train_icd_num'] >100) & (icd_concat['train_icd_num'] <= 1000)][model + '_auc'].mean())
        ROC_AUC.append(icd_concat[(icd_concat['train_icd_num'] >1000) & (icd_concat['train_icd_num'] <= 10000)][model + '_auc'].mean())
        ROC_AUC.append(icd_concat[(icd_concat['train_icd_num'] >10000) ][model + '_auc'].mean())
        icd_results[model+'_auc'] = ROC_AUC

        F1.append(icd_concat[icd_concat['train_icd_num'] == 0][model + '_f1'].mean())
        F1.append(icd_concat[(icd_concat['train_icd_num'] > 0) & (icd_concat['train_icd_num'] <= 10)][model + '_f1'].mean())
        F1.append(icd_concat[(icd_concat['train_icd_num'] > 10) & (icd_concat['train_icd_num'] <= 100)][model + '_f1'].mean())
        F1.append(icd_concat[(icd_concat['train_icd_num'] > 100) & (icd_concat['train_icd_num'] <= 1000)][model + '_f1'].mean())
        F1.append(icd_concat[(icd_concat['train_icd_num'] > 1000) & (icd_concat['train_icd_num'] <= 10000)][model + '_f1'].mean())
        F1.append(icd_concat[(icd_concat['train_icd_num'] > 10000)][model + '_f1'].mean())
        icd_results[model + '_f1'] = F1

    icd_results.to_csv(os.path.join(output_dir, args.model_name+"_mean.csv"))
    # np.save(os.path.join(output_dir,'rareseeker_negativesampling_fpr.npy'),fpr)
    # np.save(os.path.join(output_dir,'rareseeker_negativesampling_tpr.npy'),tpr)
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_AUROC.npy'),roc_auc)
    # np.save(os.path.join(output_dir,'rareseeker_negativesampling_precision.npy'), precision_)
    # np.save(os.path.join(output_dir,'rareseeker_negativesampling_recall.npy'),recall_)
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_AUPRC.npy'),AUPRC)
    np.save(os.path.join(output_dir,'rareseeker_negativesampling_f1.npy'),f1)


    return results


def find_threshold(val_outputs, val_label, threshold):
    val_outputs = torch.sigmoid(val_outputs).data.numpy()

    for i in threshold.keys():
        y_val = val_label[:, i]
        val_score = val_outputs[:, i]
        if np.sum(y_val) == 0:
            threshold[i] = np.median(val_score)
        else:
            sort_score = np.argsort(val_score)
            sort_label = np.take_along_axis(y_val, sort_score, axis=0)
            label_count = np.sum(sort_label)
            TP = label_count - np.cumsum(sort_label)  # TP
            TN = np.cumsum(sort_label == 0)
            # accuracy = (TP+TN)/len(y_val)
            # acc_argmax = np.argmax(accuracy)
            # score_ascending = np.take_along_axis(val_score, sort_score, axis=0)
            # best_threshold = score_ascending[acc_argmax]
            FP = np.cumsum(sort_label)
            FN = len(y_val) - label_count - np.cumsum(sort_label == 0)
            f1 = (2 * TP) / (2 * TP + FP + FN)
            f1_argmax = np.argmax(f1)
            score_ascending = np.take_along_axis(val_score, sort_score, axis=0)
            best_threshold = score_ascending[f1_argmax]
            # print("threshold:", best_threshold, "max_acc:", np.max(accuracy))
            threshold[i] = best_threshold
    return threshold


def find_threshold_micro(val_outputs_nonzero, val_label_nonzero):
    y_val = val_label_nonzero.flatten()
    val_outputs = val_outputs_nonzero.flatten()
    val_score = torch.sigmoid(val_outputs).data.numpy()
    sort_score = np.argsort(val_score)
    sort_label = np.take_along_axis(y_val, sort_score, axis=0)
    label_count = np.sum(sort_label)
    TP = label_count - np.cumsum(sort_label)
    FP = np.cumsum(sort_label)
    FN = len(y_val) - label_count - np.cumsum(sort_label == 0)
    f1 = (2 * TP) / (2 * TP + FP + FN)
    f1_argmax = np.argmax(f1)
    score_ascending = np.take_along_axis(val_score, sort_score, axis=0)
    best_threshold = score_ascending[f1_argmax]
    # print("threshold:", best_threshold, "max_acc:", np.max(accuracy))
    threshold_micro = best_threshold

    return threshold_micro


def find_threshold_macro(val_outputs, val_label, threshold_macro):
    val_outputs = torch.sigmoid(val_outputs).data.numpy()

    for i in threshold_macro.keys():
        y_val = val_label[:, i]
        val_score = val_outputs[:, i]
        if np.sum(y_val) == 0:
            threshold_macro[i] = np.median(val_score)
        else:
            sort_score = np.argsort(val_score)
            sort_label = np.take_along_axis(y_val, sort_score, axis=0)
            label_count = np.sum(sort_label)
            TP = label_count - np.cumsum(sort_label)
            FP = np.cumsum(sort_label)
            FN = len(y_val) - label_count - np.cumsum(sort_label == 0)
            f1 = (2 * TP) / (2 * TP + FP + FN)
            f1_argmax = np.argmax(f1)
            score_ascending = np.take_along_axis(val_score, sort_score, axis=0)
            best_threshold = score_ascending[f1_argmax]
            # print("threshold:", best_threshold, "max_acc:", np.max(accuracy))
            threshold_macro[i] = best_threshold
    return threshold_macro


def main(args):
    model_checkpoint_loc = os.path.join(args.save_dir, "best_classifier.pth.tar")
    output_dir = os.path.join(args.save_dir, "test")
    if args.debug == 1:
        output_dir = os.path.join(args.save_dir, "debug")

    use_gpu = args.use_gpu
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    model_name = model_checkpoint_loc.split('/')[-1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("The test directory doesn't exist so it is created.")

    train_label,val_feature,val_label, test_feature,test_label,label_emb = load_data(args)
    print('Done loading data.')

    test(train_label, val_feature, test_feature, val_label, test_label, label_emb,
         model_checkpoint_loc, output_dir, use_gpu, hidden_size, batch_size)


if __name__ == "__main__":
    time0 = time.time()
    args = args_argument()
    main(args)
    print('Time used', str(time.time() - time0))
