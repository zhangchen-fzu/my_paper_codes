from knrm_xgboost.src.knrm_xgboost import KNRM_XGBOOST
import pickle
import pandas as pd
import itertools
import numpy as np

##训练集测试集词语形成一个词嵌入词典
print('create dict ...')
words=[]
train=pd.read_excel(r'D:\my paper aa\train\train_trans_labeled_token1.xlsx')
for i in range(len(train)):
    word1=train['zh_term'][i].split()
    words.append(word1)
    word2=train['es_term'][i].split()
    words.append(word2)
test=pd.read_csv(r'D:\my paper aa\test\test_trans_labeled_token.csv')
for i in range(len(test)):
    word1 = test['zh_term'][i].split()
    words.append(word1)
    word2 = test['es_term'][i].split()
    words.append(word2)
words=list(set(list(itertools.chain.from_iterable(words))))
word_vector_dic={}
with open(r'g:\my paper\预对齐嵌入向量\all_align_quchong.vec') as f:
    for each_vec in f:
        vector=each_vec.split(' ')
        if vector[0] in words:
            v=vector[1:]
            v1=[]
            for i in range(len(v)):
                v1.append(float(v[i]))
            word_vector_dic[vector[0]]=np.array(v1)
for i in range(len(words)):
    if words[i] not in list(word_vector_dic.keys()):  ##没有找到对应的向量，则赋值[-0.2,0.2]之间的随机值
        initializer = np.random.uniform(-0.2, 0.2)
        word_vector_dic[words[i]]=[np.array([initializer]*300)]
print('dic created ...')

###开始训练
print("train start ...")
kx = KNRM_XGBOOST(word_vector_dic)
kx.trian(r'D:\my paper aa\train\train_trans_labeled_token1.xlsx', r'd:\my paper aa\knrm_xgboost.model')
print("train end ...")

# ##开始预测
model = pickle.load(open('d:\my paper aa\knrm_xgboost.model', 'rb'))
kx.test(r'd:\my paper aa\test\test_trans_labeled_token.csv', model)
# # 82.34%