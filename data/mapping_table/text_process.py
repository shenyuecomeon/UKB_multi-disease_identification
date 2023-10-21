import pandas as pd
import numpy as np
import ipdb
import re
import seaborn as sns
import matplotlib.pyplot as plt

# text_new1 = pd.read_csv('text_new1.csv')
# text_new1['word count'] = text_new1.iloc[:, 1].apply(lambda x: len(x.split()))
# bins = list(range(0, 440, 40))
# labels = [f'{x}-{x+40}' for x in range(0, 400, 40)]
# text_new1['category'] = pd.cut(text_new1['word count'], bins=bins, labels=labels)
# counts = text_new1['category'].value_counts()
#
# sns.set()
# sns.distplot(text_new1['word count'], kde=False)
# plt.savefig('/home/yuis/POPDx/process_5_4/plot/test.png', dpi=300)
#
# print(text_new1['word count'].max())

def join_strings(row):
    if all(val == '' for val in row):
        return '[SEP] '
    else:
        non_empty_values = [str(val) for val in row if val != '']
        return ' '.join(non_empty_values) + ' [SEP] '

def lowercase_except(text):
    special_words = ['SEP', 'NEC', 'ENT', 'NOC', 'NOS', 'BCR', 'ABL', 'PTLD', 'AHP']
    pattern = r"\b(?!{}\b)\w+".format("|".join(special_words))
    lowercase_text = re.sub(pattern, lambda x: x.group().lower(), text)
    return lowercase_text

# fields = ["41210","41200","20002","20004","40011","41246","20003","22601"] #5_26版本
#fields = ["41210","41200","20003","22601","20002","40011","20004","41246"] #5_20版本
fields = ["41210","41200","22601","40006","40013","20002","20001","20004","40011","41246","41245","20003"]
# text_new3修改文本的前后顺序变成5_20这一版本
ukb_cols = pd.read_csv("/home/shared_data/zdukb/basic/CSV/ukb47147.csv",nrows=1).columns

my_dict = {key:None for key in fields}

for i in fields:
    my_dict[i] = [k for k in ukb_cols if k.split("-")[0]==i]
field_cols = [i for j in my_dict.values() for i in j ]
# ukb_text = pd.read_csv("/home/shared_data/zdukb/basic/CSV/ukb47147.csv",usecols=["eid"]+field_cols).set_index("eid")
record_coding = pd.read_excel("/home/shared_data/zdukb/POPDx/coding/record_coding_new.xlsx",sheet_name=["Sheet2"])["Sheet2"]
record_dict = dict(zip(record_coding['field'], record_coding['coding']))
df_ = []

for f in fields:
    sn = record_dict[int(f)].split('.')[0]
    df_map = pd.read_excel("/home/shared_data/zdukb/POPDx/coding/merged_file.xlsx",sheet_name=[sn])[sn]
    mapping_dict = dict(zip(df_map['coding'],df_map['meaning']))
    chunk = pd.read_csv("/home/shared_data/zdukb/basic/CSV/ukb47147.csv", usecols=["eid"] + my_dict[f]).set_index("eid")
    chunk = chunk.applymap(lambda x: mapping_dict.get(x, np.nan))
    chunk.fillna('', inplace=True)
    # chunk[f+'-new'] = chunk.apply(lambda row: ''.join(str(val) for val in row)+' [SEP] ', axis=1)
    chunk[f + '-new'] = chunk.apply(join_strings, axis=1)
    if f == '20003':
        chunk[f + '-new'] = chunk[f + '-new'].str.replace('s/f', 'sugar free')
        chunk[f + '-new'] = chunk[f + '-new'].str.replace('m/r', 'modified release')
        chunk[f + '-new'] = chunk[f + '-new'].str.replace('e/c', 'enteric coated')
        chunk[f + '-new'] = chunk[f + '-new'].str.replace('eye/ear', 'eye or ear')
        # if chunk[f + '-new'].str.contains(r'[a-zA-Z]/[a-zA-Z]').any():
        #     print('still')
        chunk[f + '-new'] = chunk[f + '-new'].str.replace('/', ' per ')

    df_.append(chunk[f+'-new'].copy().to_frame())

df_ = pd.concat(df_,axis=1)



# 将每一行的所有字符串拼接在一起
df_ = df_.apply(lambda x: ''.join(x.astype(str)), axis=1)

# 将连续超过两个 [SEP] 的字符串替换为一个 [SEP]
# df_ = df_.apply(lambda x: re.sub(r'\s*\[SEP\]\s*(\[SEP\]\s*)+', ' [SEP] ', x))
df_ = df_.apply(lambda x: re.sub(r'\s+\[SEP\]\s*(\[SEP\]\s*)+', ' [SEP] ', x))
df_ = df_.apply(lambda x: re.sub(r'\[SEP\]\s*(\[SEP\]\s*)+', '[SEP] ', x))
# df_ = df_.apply(lambda x: re.sub(r'\b[a-zA-Z]\d+\b', '', x))
# df_ = df_.apply(lambda x: re.sub(r'\b[a-zA-Z]\d+(\.\d+)?\b', '', x))
df_ = df_.apply(lambda x: re.sub(r'\b[A-Z]\d+\.\d+\s\b', '', x))
df_ = df_.apply(lambda x: re.sub(r'\+/-', 'with or without', x))
df_ = df_.apply(lambda x: re.sub(r'\s*/\s*', ' or ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*&\s*', ' and ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*<\s*', ' less than ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*>\s*', ' more than ', x))
df_ = df_.apply(lambda x: re.sub(r'\(', '', x))
df_ = df_.apply(lambda x: re.sub(r'\)', '', x))
df_ = df_.apply(lambda x: re.sub(r'%', ' percent', x))
df_ = df_.apply(lambda x: re.sub(r'\s*\+\s*', ' and ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*\[(?!SEP\])', " [ ", x))
df_ = df_.apply(lambda x: re.sub(r'(?<!\[SEP)\]', " ]", x))
df_ = df_.apply(lambda x: re.sub(r'\s*:\s*', ' : ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*,\s*', ' , ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*\.\s*', ' . ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*-\s*', ' - ', x))
df_ = df_.apply(lambda x: re.sub(r'\s*\'\s*', ' \' ', x))
pattern = re.compile('[\n]|【|】|([^\u0000-\u00FF\u1100-\u11FF\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3130-\u318F\u3400-\u4dbf\u4e00-\u9faf\uAC00-\uD7AF\uff00-\uffef])')
df_ = df_.apply(lambda x: pattern.sub(' ',x))
cleanr = re.compile('NaN')
# df_ = df_.apply(lambda x: cleanr.sub(' ', data['product_title'][i].replace('NaN', '')))

df_ = df_.apply(lowercase_except)
# df_ = df_.replace("[sep]", "<s>", regex=True)
# , . - '
# | @ _ ! $ = { } * ? "
df_ = df_.apply(lambda x: re.sub(r"\[SEP\]", "<s>", x))
df_.to_csv("text_field12.csv")
ipdb.set_trace()
# df_.to_csv("text_new3.csv")

ipdb.set_trace()
# phecode_csv = pd.read_csv("../process_5_4/phecode_csv.csv")
# mapping_dict = dict(zip(phecode_icd['ALT_CODE'], phecode_icd['PheCode']))
# mapping_dict2 = dict(zip(phecode_csv.iloc[:,1], phecode_icd.iloc[:,0]))
# df = pd.read_csv("final_icd_match.csv").set_index("eid")
# mapped_df = df.applymap(lambda x: mapping_dict.get(x, np.nan))
# mapped_df = mapped_df.applymap(lambda x: mapping_dict2.get(x, np.nan))
# df_date =  pd.read_csv("final_date_matchchunk[f+'-new'].copy().to_frame()).csv").set_index("eid")
# mask = mapped_df.notnull().rename(columns=dict(zip(df.columns, df_date.columns)))
# df_date =df_date.where(mask)
# print("icd里有%dnan"%(sum(df.isnull().sum())))
# print("phecode里有%dnan"%(sum(mapped_df.isnull().sum())))













