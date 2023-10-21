import torch
import torch.nn as nn
import time
from transformers import BertConfig, AutoTokenizer
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import BertPrefixForMLclassification_conv

from data import *
from tools import ModelSaving
from logger import *
from torch.autograd import Variable

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
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Default batch size is 513.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Default learning rate is 0.0001')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.000, help='Default weight decay is 0')
    parser.add_argument('-max_len', '--max_length', type=int, default=512, help='Default max length is 64')
    parser.add_argument('-bh', '--bert_hidden_size', type=int, default=768, help='Default max length is 64')
    parser.add_argument('--use_gpu', default=True, help='Default setup is to not use GPU for test.')
    parser.add_argument('--debug', type=int, default=None)

    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--net",type=str,default="conv")
    parser.add_argument('--self_loop',type = int, default = 2)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument('--classifi',type=str,default="icd")
    parser.add_argument('--try1',type=int,default=3)
    parser.add_argument('--try2',type=int,default=0)
    parser.add_argument('--lower_weight', type=float, default=0.9)
    parser.add_argument('--upper_weight', type=float, default=0.7)
    parser.add_argument('--pretrain',type= int,default=0)
    args = parser.parse_args()
    return args    

def generate_cv_icd(args):
    adj = np.load('./embeddings/adj.npy',allow_pickle=True)

    I = args.self_loop*np.eye(adj.shape[0])
        
    k = args.alpha
    
 
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
    

    adj = np.load('./onto_embeddings/adj_onto.npy',allow_pickle=True)

    invsqrt_degree_matrix = np.diag(1/np.sqrt(np.sum(adj, axis=1)))
    A = invsqrt_degree_matrix@adj@invsqrt_degree_matrix    
    conv = I+ k*A+ k*k*A@A + k*k*k*A@A@A 
    
    phecode_labels = pd.read_csv("/home/shared_data/zdukb/COVID-19/process_5_4/phecode_csv.csv")["PHECODE"].tolist()
    
    with open("./onto_embeddings/embedding_node_to_idx_dict.json", 'r') as inpt:
        l2i = json.load(inpt)
    labels_idx = [l2i[str(delete_extra_zero(l))] for l in phecode_labels]
    conv= conv[labels_idx,:][:,labels_idx]
    print("done generating convolution matrix")
    return conv

def load_data(logger):
    if args.classifi == "icd":
        data_folder = '/home/shared_data/zdukb/POPDx/data_new/icd'
        train_feature_file = 'train_feature_16.npy' 
        train_label_file = 'train_icd_labels.npy'
        val_feature_file = 'val_feature_16.npy'
        val_label_file = 'val_icd_labels.npy'
        label_emb = generate_cv_icd(args)
        train_text_file = '/home/shared_data/zdukb/POPDx/data_new/icd/train_text_raw12.txt'
        val_text_file = '/home/shared_data/zdukb/POPDx/data_new/icd/val_text_raw12.txt'
    else:
        data_folder = '/home/shared_data/zdukb/POPDx/data_new/phecode'
        train_feature_file = 'train_feature_16.npy' 
        train_label_file = 'train_phe_labels.npy'
        val_feature_file = 'val_feature_16.npy'
        val_label_file = 'val_phe_labels.npy'
        label_emb = generate_cv_phe(args)
        train_text_file = '/home/shared_data/zdukb/POPDx/data_new/phecode/train_text_raw12_phe.txt'
        val_text_file = '/home/shared_data/zdukb/POPDx/data_new/phecode/val_text_raw12_phe.txt'

    train_feature = np.load(os.path.join(data_folder, train_feature_file), allow_pickle=True)
    val_feature = np.load(os.path.join(data_folder, val_feature_file), allow_pickle=True)
    train_label = np.load(os.path.join(data_folder, train_label_file))
    val_label = np.load(os.path.join(data_folder, val_label_file))
    
    logger.write_line(
        "====================================== DATA SUMMARY ======================================", True)
    logger.write_line("Train Feature File:" + train_feature_file, True)
    logger.write_line("Val Feature File:" + val_feature_file, True)

    return train_feature, val_feature, train_label, val_label, label_emb,train_text_file,val_text_file


def train(train_feature, val_feature, train_label, val_label, label_emb,train_text_file,val_text_file,
          use_cuda=True, hidden_size=150, learning_rate=0.0001, weight_decay=0.05, save_dir='', logger=None, args = None):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    with open(train_text_file, "r", encoding='utf-8') as file:
        train_text = [line.strip() for line in file.readlines()]

    with open(val_text_file, "r", encoding='utf-8') as file:
        val_text = [line.strip() for line in file.readlines()]
    if args.debug == 1:   

        train_feature = train_feature[:2000*args.batch_size, :]
        val_feature = val_feature[:2000*args.batch_size,:]

        with open(train_text_file, "r", encoding='utf-8') as file:
            train_text = [line.strip() for line in file.readlines()[:2000*args.batch_size]]

        with open(val_text_file, "r", encoding='utf-8') as file:
            val_text = [line.strip() for line in file.readlines()[:2000*args.batch_size]]


    label_emb = torch.tensor(label_emb, dtype=torch.float, device=device)

    def collate_fn(batch):
        # 将句子列表和NumPy数据分离
        text = [item["text"] for item in batch]
        x_data = torch.stack([item["x_data"] for item in batch])
        y_data = torch.stack([item["y_data"] for item in batch])
        
        # 使用BERT分词器对句子进行分词和编码
        encoded_sentences = biotokenizer(text, padding="max_length", max_length = args.max_length-args.pre_seq_len, truncation=True, return_tensors="pt",  add_special_tokens = True)
        return {
            "input_ids": encoded_sentences["input_ids"],
            "attention_mask": encoded_sentences["attention_mask"],
            "x_data": x_data,
            "y_data": y_data
        }
    
    biotokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")
    if args.pretrain == 1:
        biotokenizer = AutoTokenizer.from_pretrained("/home/shared_data/zdukb/POPDx/save_new/phe_7_22_pretrain")
    train_dataset = Dataset_pipline(train_feature, train_label, train_text)
    val_dataset = Dataset_pipline(val_feature, val_label, val_text)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    logger.write_line(
        "====================================== TRAIN SUMMARY ======================================", True)
    logger.write_line("Train Set:\t" + train_loader.dataset.__class__.__name__, True)
    logger.write_line("Val Set:\t" + val_loader.dataset.__class__.__name__, True)
    
    logger.write_line("-Train set length:\t" + str(len(train_loader)), True)
    logger.write_line("-Val set length:\t" + str(len(val_loader)), True)
    logger.write_line("-Train Batch size:\t" + str(args.batch_size), True)
    logger.write_line("-Val Batch size:\t" + str(args.batch_size), True)
    logger.write_line("Label embedding shape:" + str(label_emb.shape), True)
    logger.write_line("Train feature shape:" + str(train_feature.shape), True)
    
    config = BertConfig.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    net = BertPrefixForMLclassification_conv(config,train_feature.shape[1],hidden_size,label_emb,args.pre_seq_len)
    net.initialize()
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    n_epochs = 50
    early_break = ModelSaving(waiting=5, printing=True)
    train = []
    val = [] 
    val_lowest = np.inf
    save_dir = save_dir

    logger.write_line(
        "====================================== TRAIN RECORD ======================================", True)
    for epoch in range(n_epochs):
        # training the model 
        logger.write_line('starting epoch ' + str(epoch), True)
        net.train()
        losses = []
        progress_bar = tqdm(range(len(train_loader)))
        
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            
            train_inputs, train_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
            train_inputs.requires_grad_()
            # ipdb.set_trace()
            if (args.net == "onlytext") or (args.net == "conv_onlytext") or (args.net == "convnn_onlytext"):
                train_outputs = net(input_ids, attention_mask)
            else:
                train_outputs = net(train_inputs, input_ids, attention_mask)
            loss = criterion(train_outputs, train_labels)
            optimizer.zero_grad()
            # ipdb.set_trace()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
            #print('[%d/%d] Training Loss: %.3f' % (epoch+1, batch_idx, loss))
            progress_bar.update(1)

         
        logger.write_line('[%d/%d] Training Loss: %.3f' % (epoch + 1,batch_idx, loss), True)
        print('[%d/%d] Training Loss: %.3f' % (epoch+1,batch_idx,loss))
        # validating the model
        net.eval()  
        val_losses = []
                
        progress_bar2 = tqdm(range(len(val_loader)))
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            # ipdb.set_trace()
            val_inputs, val_labels, input_ids, attention_mask = Variable(batch["x_data"].to(device)), Variable(batch["y_data"].to(device)), Variable(batch["input_ids"].to(device)), Variable(batch["attention_mask"].to(device))
 
            val_outputs = net(val_inputs, input_ids, attention_mask)
            val_loss = criterion(val_outputs, val_labels)
            val_losses.append(val_loss.data.mean().item())
            #print('[%d/%d] Validation Loss: %.3f' % (epoch+1, batch_idx, val_loss))
            progress_bar2.update(1)
        logger.write_line('[%d] Validation Loss: %.3f' % (epoch + 1,val_loss), True)

        print('[%d] Validation Loss: %.3f' % (epoch+1, val_loss))
        train.append(losses)
        val.append(val_losses)
        if np.mean(val_losses) < val_lowest:
            val_lowest = np.mean(val_losses)
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': np.mean(losses),
                            'val_loss': np.mean(val_losses)
                            }, os.path.join(save_dir, 'best_classifier.pth.tar'))
            torch.save(net.bert.state_dict(), os.path.join(save_dir,'best_bert.pth'))
            logger.write_line(str(val_lowest)+' saved', True)
        early_break(np.mean(val_losses), net)

        if early_break.save:
            print("Maximum waiting reached. Break the training.")
            break
                            
    train_L = [np.mean(x) for x in train]
    val_L = [np.mean(x) for x in val]

    plt.plot(list(range(1,len(train_L)+1)), train_L,'-o',label='Train')
    plt.plot(list(range(1,len(val_L)+1)), val_L,'-x',label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(save_dir,'train_loss.png'))


def main(args):
    
    model_checkpoint_loc = args.save_dir
                            
    if not os.path.exists(model_checkpoint_loc):
        os.makedirs(model_checkpoint_loc)
        print("The save directory doesn't exist so it is created.")
    use_gpu = args.use_gpu
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate 
    weight_decay =  args.weight_decay       
    
    logger = Logger(model_checkpoint_loc)
    logger.initialize_file("train")
    logger.write_line(str(args), True)


    train_feature, val_feature, train_label, val_label, label_emb,train_text_file,val_text_file = load_data(logger)
    train(train_feature, val_feature, train_label, val_label, label_emb, train_text_file,val_text_file,
          use_cuda=True, hidden_size=hidden_size, 
          learning_rate=learning_rate, weight_decay=weight_decay,           
          save_dir=model_checkpoint_loc, logger=logger, args=args)
    return logger
    
if __name__ =="__main__":
    time0 = time.time()
    args = args_argument()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = main(args)
    print('Time used', str(time.time() - time0))         
    logger.write_line('Time used', str(time.time() - time0))
