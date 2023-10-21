import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import *
import random
import ipdb
## process labels
random.seed(1)
df_icd = pd.read_csv('/home/shared_data/zdukb/basic/ukb47147_icd10.csv').set_index("eid")
# ipdb.set_trace()
# df_icd = df_icd.dropna(axis=0, how="all") #删除全是空值的行
non_train_eid = []
for id in df_icd.index:
    if df_icd.loc[id,:].isnull().sum()>220:
        non_train_eid.append(id)
ipdb.set_trace()
df_map = pd.read_csv('../data/Phecode_map_icd10.csv')
temp = list(set(df_icd.index)-set(non_train_eid))
non_train_eid += random.sample(temp,int(len(temp)*0.2))
print(len(non_train_eid)) #从中划分测试集和验证集
train_eid = list(set(df_icd.index)-set(non_train_eid))
print(len(train_eid))


# with open('../data/phecode_labels.txt', 'r') as f:
#     phecode_labels = f.readlines() 
# phecode_labels = pd.Series(["{:0.2f}".format(float(x.strip())) for x in phecode_labels])

# labels_onehot = np.zeros((df_icd.shape[0], len(phecode_labels)))

# for i in range(len(df_icd)):
#     phecode = df_map.PHECODE[df_map.ICD10.isin(df_icd.iloc[i, 2:])].to_numpy()
#     labels_onehot[i, :] = phecode_labels.isin(["{:0.2f}".format(x) for x in phecode]).to_numpy().T * 1
# ipdb.set_trace()

# np.save('../data/labels_onehot.npy', labels_onehot)
# df_icd.eid.to_csv('../data/eid_labels.csv')

## split the data into training and test set
labels_onehot = np.load("../data/labels_onehot.npy")
eid_labels = pd.read_csv("../data/eid_labels.csv")

#num_sparse = np.load("../data/x_data_2023_2_20/x_data_2023_2_20/num_data_sparse.npz")
cate_sparse = np.load("../data/x_data_2023_2_20/cate_data_sparse1.npz")
recover_num_data = np.load("../data/x_data_2023_2_20/num_data.npy")
recover_cate_data = csc_matrix((cate_sparse['data'], cate_sparse['indices'], cate_sparse['indptr'])).toarray()
print('Finish the transformation of the data!')

ipdb.set_trace()
features = np.concatenate([recover_num_data, recover_cate_data, labels_onehot],axis=1)
df_all = pd.DataFrame(features)
df_all.index = eid_labels["eid"]
df_train = df_all[df_all.index.isin(train_eid)]
df_test_val = df_all[df_all.index.isin(non_train_eid)]
# assert len(df_train)+len(df_test_val) == len(df_all)
print(df_train)
print('Finish the combination of the data!')



train_array = np.asarray(df_train)
X_train, Y_train = train_array[:,0:-labels_onehot.shape[1]], train_array[:,-labels_onehot.shape[1]:]
print(X_train.shape, Y_train.shape)

X_test, X_val, Y_test, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

## save the results
np.save('../data/data_3_19/train_feature.npy', X_train)
np.save('../data/data_3_19/train_phecode_labels.npy', Y_train)
np.save('../data/data_3_19/val_feature.npy', X_val)
np.save('../data/data_3_19/val_phecode_labels.npy', Y_val)
np.save('../data/data_3_19/test_feature.npy', X_test)
np.save('../data/data_3_19/test_phecode_labels.npy', Y_test)
