# 随机将数据分成训练集10000词，4800测试集，4800开发集
# import pandas as pd
# import itertools
# data1=pd.read_csv(r'D:\my paper xiugai\人工标记\candidate_termpair_rank_quchong_labels0_labeled.csv',usecols=[0],encoding='utf-8').values.tolist()
# data1=list(itertools.chain.from_iterable(data1))
# word=list(set(data1))
# data=pd.read_csv(r'D:\my paper xiugai\人工标记\candidate_termpair_rank_quchong_labels0_labeled.csv',names=['zh_term','es_term','trans','label'],encoding='utf-8')
#
# val=[]
# test=[]
# for i in range(len(word)):
#     if  i<1320:
#         val.append(word[i])
#     if i>=1320 and i<2640:
#         test.append(word[i])
#
# for j in range(len(val)):
#     pos_samples = data[data.zh_term == val[j]]
#     pos_samples.to_csv(r'D:\my paper xiugai\人工标记\candidate_termpair_rank_quchong_labels0_labeled_val.csv', index=False, header=0, mode='a')
# print("2")
#
# for k in range(len(test)):
#     pos_samples = data[data.zh_term == test[k]]
#     pos_samples.to_csv(r'D:\my paper xiugai\人工标记\candidate_termpair_rank_quchong_labels0_labeled_test.csv', index=False, header=0, mode='a')
#
# print("训练集、测试集、开发集随机分离完毕！")

##随机合成训练集5个负样本
# import pandas as pd
# import random
# import itertools
# data1=pd.read_csv(r'd:\my paper aa\新随机合成训练集\train_trans_labeled0.csv',usecols=[1]).values.tolist()
# data1_es_term=list(itertools.chain.from_iterable(data1))
# wd2idx={}
# for i in range(len(data1_es_term)):
#     wd2idx[i]=data1_es_term[i]
# # print(wd2idx)
# data3=pd.read_excel(r'd:\my paper aa\新随机合成训练集\联合国语料库改删排筛.xlsx')
# data=pd.read_excel(r'd:\my paper aa\新随机合成训练集\联合国语料库改删排筛.xlsx',usecols=[0]).values.tolist()
# data_zh_term=list(set(list(itertools.chain.from_iterable(data))))
# for i in range(len(data_zh_term)):
#     group_data=data3[data3.zh_term==data_zh_term[i]]
#     group_data = group_data.reset_index()
#     group_data = group_data.drop(columns='index')
#     zh_term=[]
#     es_term=[]
#     biaoqian=[]
#     index = random.sample(range(0, len(wd2idx)), 5)
#     for m in range(5):
#         zh_term.append(data_zh_term[i])
#         biaoqian.append(0)
#     for i in range(len(index)):
#         es_term.append(wd2idx[index[i]])
#     zh_term=pd.DataFrame(zh_term,columns=['zh_term'])
#     es_term=pd.DataFrame(es_term,columns=['es_term'])
#     biaoqian=pd.DataFrame(biaoqian,columns=['biaoqian'])
#     data_new=pd.concat([zh_term,es_term,biaoqian],axis=1)
#     data_new1=pd.concat([group_data,data_new],axis=0)
#     data_new1=data_new1.sample(frac=1,random_state=None)
#     data_new1.to_csv(r'd:\my paper aa\新随机合成训练集\联合国语料库改删排筛负随机.csv',header=0, mode='a', index=False)


