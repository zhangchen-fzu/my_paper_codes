import pandas as pd
import itertools
import numpy as np
# np.random.seed(27)

words = []
train = pd.read_excel(r'D:\my paper aa\train\train_trans_labeled_token2.xlsx')
for i in range(len(train)):
    word1 = train['zh_term'][i].split()
    words.append(word1)
    word2 = train['es_term'][i].split()
    words.append(word2)
test = pd.read_csv(r'D:\my paper aa\test_new\test_set1.csv')
for i in range(len(test)):
    word1 = test['zh_term'][i].split()
    words.append(word1)
    word2 = test['es_term'][i].split()
    words.append(word2)
words = list(set(list(itertools.chain.from_iterable(words))))
word_vector_dic = {}
with open(r'G:\my paper\pre_enb_vector\all_align_quchong.vec', encoding='utf-8') as f:  ##当使用了PART向量之后会出现问题，每行有一个'\n'，要去掉
    for each_vec in f:
        vector = each_vec.split(' ')
        if vector[0] in words:
            v = vector[1:]
            v1 = []
            for i in range(len(v)):
                v1.append(float(v[i]))
            word_vector_dic[vector[0]] = np.array(v1)
for i in range(len(words)):
    if words[i] not in list(word_vector_dic.keys()):  ##没有找到对应的向量，则赋值[-0.2,0.2]之间的随机值
        initializer = np.random.uniform(-0.2, 0.2)
        word_vector_dic[words[i]] = np.array([initializer] * 300)

test=pd.read_csv(r'D:\my paper aa\test_new\test_set1.csv')
sims=[]
for i in range(len(test)):
    zh_term=[]
    for j in test['zh_term'][i].spilt():
        a=list(word_vector_dic[j])
        zh_term.append(a)
    total_zh_term = [sum(x) for x in zip(*zh_term)]
    evg_total_zh_term = [total_zh_term[i] / len(zh_term) for i in range(len(total_zh_term))]
    es_term=[]
    for k in test['es_term'][i].spilt():
        b=list(word_vector_dic[k])
        es_term.append(b)
    total_es_term=[sum(x) for x in zip(*es_term)]
    evg_total_es_term = [total_es_term[i] / len(es_term) for i in range(len(total_es_term))]

    evg_total_zh_term = np.array(evg_total_zh_term)
    evg_total_es_term = np.array(evg_total_es_term)
    sim = (np.matmul(evg_total_zh_term, evg_total_es_term)) / ((np.linalg.norm(evg_total_zh_term) )*(np.linalg.norm(evg_total_es_term)))
    sims.append(sim)
sims=pd.DataFrame(sims,columns=['cos_sim'])
sims.to_csv(r'D:\my paper aa\cos_sim_result\cos_sim.csv',index=False,encoding='utf-8')

