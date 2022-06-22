import numpy as np
import pandas as pd
import re
import os
import io
# a=['scsdcs','sdc','gfb']
# b=' '.join(a)
# print(b)
# import pandas as pd
# datas=pd.read_excel(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela1.xlsx',encoding='utf-8')
# lens=[]
# for i in range(len(datas)):
#     print(i)
#     lens.append(len(datas['term'][i].split()))
# # print(lens)
# import treetaggerwrapper as tter
# tagger = tter.TreeTagger(TAGLANG='es')
# tags = tagger.tag_text('aporte')
# tags2 = tter.make_tags(tags, allow_extra=True)
# # for d in tags2:
# # #     print(d,len(d))
# import pynlpir
# pynlpir.open()
# a='适当的'
# for i in pynlpir.segment(a,pos_english=False,pos_names='child'):
#     print(i)
# # import itertools
# a=[[['xa','sd'],['xa','sada']],[['axa','sada']],[['sda','wd'],['sad','sd']]]
# a=list(itertools.chain.from_iterable(a))
# print(a)
# zh_term=[]
# data=pd.read_csv(r'g:\my paper\zh\zh_term_value_dc_dr_delzero_drdcchengji.csv','r',encoding='utf-8')
# for i in range(len(data)):
#     zh_term.append(data['word'][i])
# import itertools
# data=pd.read_csv(r'g:\my paper\es\es_term_value_dc_dr_delzero_drdcchengji.csv','r',encoding='utf-8',usecols=[0]).values.tolist()
# es_term=list(itertools.chain.from_iterable(data))
# es_all_line_term=[]  #有多少句话就有多少个[]存在
# with open(r'g:\my paper\es_alltext(对alltext的还原).txt','r',encoding='utf-8') as es_file:
#     for eachline in es_file:
#         es_each_line_term=[]
#         for j in range(len(es_term)):
#             print(j)
#             if ' '+str(es_term[j])+' ' in str(eachline):
#                 es_each_line_term.append(es_term[j])
#         es_all_line_term.append(es_each_line_term)
# print("2")
# a=['sss','ss']
# b=[]
# for i in range(len(a)):
#     if a[i] not in b:
#         b.append(a[i])
# print(b)
# import pandas as pd
# a={3:['asd','dws','sda'],2:['sda','wfaa']}
# data=list(a.values())
# print(data)
# term_list = []
# for j in data:
#     term_list.extend(j)
# # print(term_list)
# df2 = pd.DataFrame(term_list)
# df2.rename(columns={0:"term"},inplace=True)
#
# wordDict={'asd':4,'dws':3,'sda':5,'wfaa':2}
# f = pd.DataFrame(pd.Series(wordDict), columns=['freq'])
# f = f.reset_index().rename(columns={'index': 'term'})
# df = f.sort_values(by=['freq'], ascending=False)
# df3=pd.merge(df2,df,on='term',how='inner')
# print(df3)
# #改进因删除而导致出现的重复现象,先定位到空行删掉空行！！！
# datas=pd.read_excel(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela1.xlsx',encoding='utf-8')
# lens=[]
# for i in range(len(datas)):
#     lens.append(len(datas['term'][i].split()))
# lens=pd.DataFrame(lens,columns=['lens'])
# datas1=pd.concat([datas,lens],axis=1)
# datas2=datas1.sort_values(by=['lens'], ascending=False)
# for i in range(len(datas2)):
#     print(i)
#     length=datas2['lens'][i]
#     for j in range(len(datas2)):
#         if datas2['lens'][j]==length:
#             if i!=j and datas2['freq'][j]!=0 and datas2['freq'][i]!=0 and datas2['term'][i]==datas2['term'][j]:
#                 datas2['freq'][i] = datas2['freq'][i] + datas2['freq'][j]
#                 datas2['freq'][j] = 0
# datas2=datas2[datas2['freq']>0]
# datas3=datas2.sort_values(by=['lens'], ascending=False)
# datas3.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_lens.csv',index=False,encoding='utf-8')
# import re
# a='aiiaii)'
# print(re.search('^[i]*\)',a))

# #TD/B/C.II/16.
# import re
# a='1321.'
# print(re.search("^\d+\.", a))
# import os
# import re
# subdirs = os.walk(r'E:\my paper\es_del_num')
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             # with open(r'E:\my paper\es_del_num\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                     for eachline in p:
#                         if 'En virtud de la Ley' in eachline:
#                             print(fn[0:-4])

#
# for i in range(0,1000):
#     if i%100==0:
#         print(i)
# import argparse
# parser = argparse.ArgumentParser(description='Wasserstein Procrustes for Embedding Alignment')
# parser.add_argument('--model_src', type=str, help='Path to source word embeddings')
# parser.add_argument('--model_tgt', type=str, help='Path to target word embeddings')
# parser.add_argument('--lexicon', type=str, help='Path to the evaluation lexicon')
# parser.add_argument('--output_src', default='', type=str, help='Path to save the aligned source embeddings')
# parser.add_argument('--output_tgt', default='', type=str, help='Path to save the aligned target embeddings')
# parser.add_argument('--seed', default=1111, type=int, help='Random number generator seed')
# parser.add_argument('--nepoch', default=5, type=int, help='Number of epochs')
# parser.add_argument('--niter', default=5000, type=int, help='Initial number of iterations')
# parser.add_argument('--bsz', default=500, type=int, help='Initial batch size')
# parser.add_argument('--lr', default=500., type=float, help='Learning rate')
# parser.add_argument('--nmax', default=20000, type=int, help='Vocabulary size for learning the alignment')
# parser.add_argument('--reg', default=0.05, type=float, help='Regularization parameter for sinkhorn')
# args = parser.parse_args()
# import io
# fin = io.open(r'c:\Users\admin\Desktop\zh_test.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')
# for i, line in enumerate(fin):
#     tokens = line.rstrip().split(' ')
#     print(tokens)
# import numpy as np
# x=np.zeros([3,2])
# x[0, :] = np.array([1,5])
# x[1, :] = np.array([3,7])
# x[2, :] = np.array([5,9])
# print(x)
# x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
# print(np.linalg.norm(x, axis=1)[:, np.newaxis]+ 1e-8)
# print(x)
# x=np.zeros([3,2])
# x[0, :] = np.array([1,5])
# x[1, :] = np.array([3,7])
# x[2, :] = np.array([2,3])
# x-=x.mean(axis=0)[np.newaxis, :]
# print(x)
# import collections
# s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
# d = collections.defaultdict(set)
# for k, v in s:
#     d[k].add(v)
# print(d)
# print(d.items())
# A=[[1,2,3],[4,5,6]]
# A=np.array(A)
# n,d=A.shape
# print(n)
# print(d)
# a = np.arange(1, 4)
# print(np.diag(a))
# print("\n")
# np.random.seed(111)
# b = np.arange(1, 10).reshape(3, 3)
# print(b)
# xt = b[np.random.permutation(3)[:2], :]
# print(xt)
# print(np.linalg.norm(b))
# P = np.ones([2, 2]) / float(2)
# print(P)
# import ot
# a=[.5, .5]
# b=[.5, .5]
# # M=[[0., 1.], [1., 0.]]
# # c=ot.sinkhorn(a, b, M, 1)
# # print(c)
# import argparse
# parser = argparse.ArgumentParser(description='Wasserstein Procrustes for Embedding Alignment')
# parser.add_argument('--seed', default=1111, type=int, help='Random number generator seed')
# parser.add_argument('--nepoch', default=5, type=int, help='Number of epochs')
# parser.add_argument('--niter', default=5000, type=int, help='Initial number of iterations')
# parser.add_argument('--bsz', default=500, type=int, help='Initial batch size')
# parser.add_argument('--lr', default=500., type=float, help='Learning rate')
# parser.add_argument('--nmax', default=20000, type=int, help='Vocabulary size for learning the alignment')
# parser.add_argument('--reg', default=0.05, type=float, help='Regularization parameter for sinkhorn')
# args = parser.parse_args()
# np.random.seed(args.seed)
# def align( lr=10., bsz=200, nepoch=5, niter=1000,nmax=10000, reg=0.05, verbose=True):
#     print(np.random.permutation(5))
# R = align( bsz=args.bsz, lr=args.lr, niter=args.niter,nepoch=args.nepoch, reg=args.reg, nmax=args.nmax)
# def o():
#     print(np.random.permutation(5))
# a=o()
# print(4//3)
# a={0:{2,3,4},2:{7,8}}
# print(list(a.keys()))
# print(list(list(a.keys())))
# b = np.arange(1, 10).reshape(3, 3)
# print(b)
# print(b[0:2,:])
# print(b[list(list(a.keys()))])
# sc2 = np.zeros(4)
# sc2[0:3]=
# b = np.arange(1, 7).reshape(2,3)
# # print(b)
# sc=np.array([1,2,3])
# a=b-sc[np.newaxis,:]
# print(b)
# print(sc)
# print(a)
# c=[]
# a=[['as','aca'],['asd','assa']]
# for i in a:
#     c.append(' '.join(i))
# # print(c)
# b = np.arange(1, 10).reshape(3, 3)
# print(b)
# print(b[:2,:])
# fin = io.open(r'c:\Users\admin\Desktop\zh_test.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')
# n, d = map(int, fin.readline().split())
# x = np.zeros([n, d])  # 20万*300的向量
# words = []  # 按顺序存放的词语
# for i, line in enumerate(fin):  # i是行号 line是每行的所有内容  =====只要前20万行=====
# #     if i >= n:
# #         break
# #     tokens = line.rstrip().split(' ')  # 每行根据空格进行切分
# #     words.append(tokens[0])  # words中存放每行的词语
# #     v = np.array(tokens[1:], dtype=float)
# #     x[i, :] = v
# # print(words)
# # print(x)
#     print(i)
#     print(line)
# import numpy as np
# import argparse
# from utils import *
# import sys
# parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')
# parser.add_argument('--sgd', action='store_true', help='use sgd')
# params = parser.parse_args()
# if params.sgd:
# #     print(params.sgd)
# # else:
# #     print("未激发")
# # a1 = np.random.choice(a=5, size=3, replace=False, p=None)
# # print(a1)
# b = np.arange(0,36).reshape(6, 6)
# print(b)
# a=b.reshape(3,3,4)
# print(a)
# print(a.sum(1))
# # print(b.sum(1))
# c = np.arange(1, 10).reshape(3, 3)
# print(b*c)
# print(np.sum(b*c))
# K=5
# a = np.array([0, 8, 0, 4, 5, 8, 8, 0, 4, 2])
# print(a[np.argpartition(a,-K)])
# # print(a[np.argpartition(a,-K)[-K:]])
# a=np.arange(20,100).reshape(2, 40)
# print(a)
# b = np.arange(0,40).reshape(2,20)
# print(b)
# print(a[np.arange(2)[:, None], b])
# sidx = np.argpartition(b, -2, axis=1)[:, -2:]
# print(sidx)
# print(sidx.flatten())
# c = b[sidx.flatten(), :]
# print(c)
# a= np.arange(3)
# import io
# a='g:\my paper\双语嵌入结果\zh_es_embeding.txt'
# f=io.open(a[:-4]+'_tt.txt','w', encoding='utf-8')
# f.write('a')
#将汉语嵌入到西语空间中
#有监督的方法
# import numpy as np
# import argparse
# from utils import *
# import sys
# from utils import *
# parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')
# parser.add_argument("--src_emb", type=str, default=r'g:\my paper\单语嵌入\cc.zh.300.vec', help="Load source embeddings")  #####
# parser.add_argument("--tgt_emb", type=str, default=r'g:\my paper\单语嵌入\cc.es.300.vec', help="Load target embeddings")   #####
# parser.add_argument('--center', action='store_true', help='whether to center embeddings or not') #只要运行时该变量有传参就将该变量设为True
# parser.add_argument("--dico_train", type=str, default=r'g:\my paper\字典\zh-es_train.txt', help="train dictionary")  #####
# parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")
# params = parser.parse_args()
# words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)  #输出目标嵌入中的词语列表 长度为20万；以及[20万,300]的矩阵
# words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)  #输出源嵌入中的词语列表及[20万,300]的矩阵
# src2tgt, lexicon_size = load_lexicon(params.dico_train, words_src, words_tgt)#返回字典类型的双语词典对应的在源目标嵌入中的索引；双语词典的长度
# nnacc = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
# print(nnacc)
# a=['我们','你们','我们','他们']
# b=['a','b','a','a']
# # import pandas as pd
# a=pd.DataFrame(a,columns=['a'])
# b=pd.DataFrame(b,columns=['b'])
# c=pd.concat([a,b],axis=1)
# print(c)
# c.drop_duplicates(inplace=True)
# print(c)
# a=[1,0,0]
# b=[0.6,0.1,0.2]
# c=list(zip(a,b))
# a='Adfs'
# print(a.lower())
# # print(c)
# print(np.array(sorted(c, key=lambda x: x[1], reverse=True)))
# import pandas as pd

# import itertools
# data=pd.read_csv(r'C:\Users\admin\Desktop\源_分词2.csv')
# small=[]
# for i in range(len(data)):
#     small.append(data['es_term'][i].lower())
# small = pd.DataFrame(small, columns=['es_term'])
#
# data1=pd.read_csv(r'C:\Users\admin\Desktop\源_分词2.csv',usecols=[0]).values.tolist()
# data1=list(itertools.chain.from_iterable(data1))
# data1=pd.DataFrame(data1,columns=['zh_term'])
#
# data2=pd.read_csv(r'C:\Users\admin\Desktop\源_分词2.csv',usecols=[2]).values.tolist()
# data2=list(itertools.chain.from_iterable(data2))
# data2=pd.DataFrame(data2,columns=['trans'])
#
# data3=pd.read_csv(r'C:\Users\admin\Desktop\源_分词2.csv',usecols=[3]).values.tolist()
# data3=list(itertools.chain.from_iterable(data3))
# data3=pd.DataFrame(data3,columns=['biaoqian'])
#
# result=pd.concat([data1,small,data2,data3],axis=1)
# result.drop_duplicates(inplace=True)
# # result=result.sort_values(by=['zh_term'],ascending=False)
# result.to_csv(r'C:\Users\admin\Desktop\源_分词_小写2.csv',index=False,encoding='utf-8')
# print("西语大小写转换完毕！")

# import pynlpir
# import itertools
# pynlpir.open()
# wordlist=[]
# data=pd.read_excel(r'C:\Users\admin\Desktop\源2.xlsx',usecols=[0]).values.tolist()
# data=list(itertools.chain.from_iterable(data))
# tokens=[]
# for i in range(len(data)):
#     token=[]
#     for j in pynlpir.segment(data[i]):
#         token.append(j[0])
#     tokens.append(' '.join(token))
# tokens=pd.DataFrame(tokens,columns=['zh_term'])
# data1=pd.read_excel(r'C:\Users\admin\Desktop\源2.xlsx',usecols=[1]).values.tolist()
# data1=list(itertools.chain.from_iterable(data1))
# data1=pd.DataFrame(data1,columns=['es_term'])
# data2=pd.read_excel(r'C:\Users\admin\Desktop\源2.xlsx',usecols=[2]).values.tolist()
# data2=list(itertools.chain.from_iterable(data2))
# data2=pd.DataFrame(data2,columns=['trans'])
# data3=pd.read_excel(r'C:\Users\admin\Desktop\源2.xlsx',usecols=[3]).values.tolist()
# data3=list(itertools.chain.from_iterable(data3))
# data3=pd.DataFrame(data3,columns=['biaoqian'])
# reslut=pd.concat([tokens,data1,data2,data3],axis=1)
# reslut.to_csv(r'C:\Users\admin\Desktop\源_分词2.csv',index=False,encoding='utf-8')
# print("中文分词完成！")

# # import csv
# # QUOTE_NONE=3
# # data = pd.read_csv(r'd:\迅雷下载\glove.6B\glove.6B.50d.txt',
# #                    sep=" ",
# #                    index_col=0,
# #                    header=None,
# #                    )
# # print(data)
# Z = np.arange(9).reshape(3,3)
# print(Z)
# print(Z.shape)
# for index in np.ndindex(Z.shape):
#     print(index, Z[index])
# matrix = np.empty((3, 5))
# print(matrix)
# initializer=lambda: np.random.uniform(-0.2, 0.2)
# for index in np.ndindex(*matrix.shape):
#     matrix[index] = initializer()
# print(matrix)
# # import csv
# data=pd.read_csv(r'C:\Users\admin\Desktop\新建文本文档.txt',sep=" ",
#                            header=None,
#                            quoting=csv.QUOTE_NONE)
# print(data)
# data.drop_duplicates(subset=[0],inplace=True,keep='first')
# data.to_csv(r'C:\Users\admin\Desktop\新建文本文档1.txt', header=None, index=False, sep=' ', mode='a')
# data=pd.read_csv(r'c:\Users\admin\Desktop\新建文本文档1.txt',
#
#                            sep=" ",
#                            header=None,
#                            quoting=csv.QUOTE_NONE)
# print(data)
#


# data=pd.read_csv(r'g:\my paper\预对齐嵌入向量\all_align.vec',
#                             sep=" ",
#                             header=None,
#                             quoting=csv.QUOTE_NONE)
# print(len(data))
# data.drop_duplicates(subset=[0],inplace=True,keep='first')
# data.to_csv(r'g:\my paper\预对齐嵌入向量\all_align_quchong.vec', header=None, index=False, sep=' ', mode='a')
# print(len(data))
# import keras
# print(keras.__version__)
# import tensorflow
# print(tensorflow.__version__)
# import matchzoo
# print(matchzoo.__version__)
# import pandas as pd
# import matchzoo as mz
# data=pd.read_excel(r'C:\Users\admin\Desktop\1.xlsx')
# data=data.rename(columns={'zh_term':'text_left','es_term':'text_right','biaoqian':'label'})
# groups=data.sort_values(by=['label'],ascending=False).groupby(['text_left'])
# pairs=[]
# for idx, group in groups:
#     labels = group.label.unique()
#     pos_samples = group[group.label == 1]
#     pos_samples = pd.concat([pos_samples] * 2)
#     # print(pos_samples)
#     neg_samples = group[group.label < 1]
#     # print(neg_samples)
#     for _, pos_sample in pos_samples.iterrows():
#         pos_sample = pd.DataFrame([pos_sample])
#         neg_sample = neg_samples.sample(5, replace=True)
#         pairs.extend((pos_sample, neg_sample))
# new_relation = pd.concat(pairs, ignore_index=True)
# print(new_relation)
# data=pd.read_excel(r'c:\Users\admin\Desktop\1.xlsx',header=None, names = ['zh_term','es_term','label'])
#
# a=['默认','国家 项目','国家','我们','歌 曲']
#
# for i in range(len(a)):
#     # if data[data.zh_term == a[i]]!=None:
#         pos_samples = data[data.zh_term == a[i]]
#         pos_samples.to_csv(r'c:\Users\admin\Desktop\2.csv', index = False, header = 0, mode ='a')
# import pandas as pd
# data=pd.read_csv(r'D:\val_trans.csv')
# data['label']=0
# data.to_csv(r'D:\val_tran.csv',index=False,encoding='utf-8')

# a=[1,0,0,1,0,0]
# # if 1 in a:
# #     print('1')
# import itertools
# import warnings
# warnings.filterwarnings("ignore")
# term=pd.read_excel(r'c:\Users\admin\Desktop\1.xlsx',usecols=[0]).values.tolist()
# term=list(itertools.chain.from_iterable(term))
# zh_terms=list(set(term))  #10000词
#
# data=pd.read_excel(r'c:\Users\admin\Desktop\1.xlsx',header=None, names = ['zh_term','es_term','tran','label'])
# for i in range(len(zh_terms)):
#     if i%1000==0:
#         print(i)
#     group_daframe=data[data.zh_term==zh_terms[i]]  #dataframe类型
#     # print(group_daframe)
#     group_daframe=group_daframe.reset_index()
#     group_daframe = group_daframe.drop(columns='index')
#     print(group_daframe)
#
#     label = []
#     for i in range(len(group_daframe)):
#         if group_daframe['zh_term'][i] == group_daframe['tran'][i]:
#             group_daframe['label'][i] = 1
#         label.append(group_daframe['label'][i])
#     print(label)

# import pandas as pd
# data=pd.read_csv(r'd:\my paper aa\train\train_trans_labeled.csv',header=None,names=['zh_term','es_term','biaoqian'])
# for i in range(len(data)):
# import numpy as np
#
# a = np.array([1, 1])
# b = np.array([1, 6])
#
# # 这个是直接用欧式距离公式直白写法
# dist = np.sqrt(np.sum(np.square(a - b)))
# # numpy自带求欧式距离方法
# dist1 = np.linalg.norm(a - b)
#
# print(dist, dist1)
# import matchzoo as mz
# # #加载数据，转换数据类型，划分训练集和测试集，定义数据
# print('data loading ...')
# train = pd.read_excel(r'/content/drive/MyDrive/app/train_file.xlsx')
# train.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
# vald=pd.read_csv(r'/content/drive/MyDrive/app/val_trans_labeled_token.csv')
# vald.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
# test=pd.read_csv(r'/content/drive/MyDrive/app/test_trans_labeled_token.csv')
# test.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
# train_pack = mz.pack(train)
# vald_pack = mz.pack(vald)
# test_pack=mz.pack(test)
# test_pack.apply_on_text(len,mode='right',rename='length_right',inplace=True,verbose=0)
# test_pack.apply_on_text(len,mode='left',rename='length_left',inplace=True,verbose=0)

# import fasttext
# import fasttext.util
# ft = fasttext.load_model(r'G:\my paper\预对齐嵌入向量\all_align_quchong.vec')
# print(ft.get_dimension())

# fasttext.util.reduce_model(ft, 100)
# print(ft.get_dimension())
# ft.save_model('cc.en.100.bin')

##预对齐向量简化
# def kernel_mu(n_kernels, manual=False):
#     if manual:
#         return [1, 0.95, 0.90, 0.85, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.80, -0.85, -0.90, -0.95]
#     mus = [1]
#     if n_kernels == 1:
#         return mus
#     bin_step = (1-(-1))/(n_kernels-1)
#     mus.append(1-bin_step/2)
#     for k in range(1, n_kernels-1):
#         mus.append(mus[k]-bin_step)
#     return mus
# print(np.array(kernel_mu(11))[np.newaxis, np.newaxis, np.newaxis, :])
# def kernel_sigma(n_kernels):
#     sigmas = [0.001]
#     if n_kernels == 1:
#         return sigmas
#     return sigmas+[0.1]*(n_kernels-1)
# print(np.array(kernel_sigma(11))[np.newaxis, np.newaxis, np.newaxis, :])
# with open(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\data\train.csv', 'r', encoding='utf-8') as f:
#     for line in f:
#         _, query, doc, label = line.replace('\n', '').split('\t')
#         print(query)
# #         print(doc)
# # #         print(label)
# b=[]
# q_seq = [[1,2,3,4,5]]
# for i in range(2):
#     a=np.asarray([0] * 5)
#     q_seq.append(a)
# b.append(q_seq)
# d_seq = [[2,3,3,4,5]]
# for i in range(2):
#     a=np.asarray([0] * 5)
#     d_seq.append(a)
# b.append(d_seq)
# # print(b)
# d=[]
# q_seq = [[6,2,4,4,5]]
# for i in range(2):
#     a=np.asarray([0] * 5)
#     q_seq.append(a)
# d.append(q_seq)
# d_seq = [[6,7,3,9,5]]
# for i in range(2):
#     a=np.asarray([0] * 5)
# #     d_seq.append(a)
# # d.append(d_seq)
# # f=np.array(b)
# # e=np.array(d)
# # print("1",f)
# # print("2",e)
# #
# mask = np.zeros((len(f), len(f[0]), len(e[0])))
# print("3",mask)
# mask[1, :np.count_nonzero(f[1]), :np.count_nonzero(e[1])] = 1
# print("4",mask)
# #
# # print("5",mask[:, :, :, np.newaxis])
# # # def remask(mask, n_gram=1):
# #     return mask[:, n_gram-1:]
# # d=remask(c, n_gram=0+1)
# # print(d)
# # # print(len(d[0]))
# # # mask=np.zeros((5, 3, 3))
# # # mask[0, :np.count_nonzero(q[b]), :np.count_nonzero(d[b])] = 1
# # from gensim.models import KeyedVectors
# # w2v = KeyedVectors.load(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\model\w2v\w2v.model')
# # # print(w2v)
# a=[]
# c = np.array([[1,2,6],[8,7,0]])
# d=np.array(([[1,0,5],[2,0,6]]))
# a.append(c)
# a.append(d)
# # print(a)
# b=np.array(a)
# # print(b)
#
#
# f=[]
# c = np.array([[0,1,2],[0,0,0]])
# d=np.array(([[6,0,0],[0,0,0]]))
# f.append(c)
# f.append(d)
# # # print(f)
# e=np.array(f)
# # print(e)
# # assert e.ndim in (2, 4)
# # print("1")
# mask = np.zeros((len(b), len(b[0]), len(e[0])))
# # print(mask)
# mask[0, :300, :1] = 1
# # print(mask)
# acc=mask[:, :, :, np.newaxis]
# print(acc)
# v=np.sum(acc, 2)
# print(v)
# w=np.log(np.clip(v, a_min=1e-10, a_max=np.inf)) * .01
# print(w,type(w))
# y=np.sum(w, 1)
# print(y,type(y))
# ab=[]
# ab.append(y)
# ab.append(y)
# print(ab,type(ab))
# print(np.concatenate(ab, axis=1))
# labels=[]
# labels.append(1)
# labels.append(0)
# labels.append(1)
# labels.append(0)
# labels.append(1)
# labels.append(0)
# print(labels)
# labels = np.array(labels)
# print(labels)
# from gensim.models import KeyedVectors
# w2v = KeyedVectors.load(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\model\w2v\w2v.model')
# print(w2v)
# a=[]
# if '真实' in w2v:
#     a.append(w2v['真实'])
# print(a,type(a))
# lines=[]
# with open(r'F:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\data\test.csv', 'r', encoding='utf-8') as f:
#     for line in f:
#         _, query, doc, label = line.replace('\n', '').split('\t')
#         lines.append(label)
# a=[]
# for i in range(len(lines)):
#     a.append(int(lines[i]))
# print(a)
# print(sum(a))

# import pandas as pd
# data=pd.read_excel(r'H:\es\1_路透社语料词语提取结果(语翼网处理后).xlsx')
# es_term=[]
# zh_term=[]
# for i in range(len(data)):
#     if data['是否成词'][i]=='是Y':
#         es_term.append(data['term'][i])
#         zh_term.append(data['翻译结果'][i])
# es_term=pd.DataFrame(es_term,columns=['es_term'])
# zh_term=pd.DataFrame(zh_term,columns=['zh_term'])
# result=pd.concat([es_term,zh_term],axis=1)
# result.to_csv(r'D:\my paper aa\train\2.csv',index=False,encoding='utf-8')

# data=pd.read_excel(r'd:\my paper aa\train\3.xlsx')
# data=data.drop_duplicates()
# data.to_csv(r'D:\my paper aa\train\4.csv')

# a=[['sc','c'],['qwd','wef']]
# import itertools
# b=list(itertools.chain.from_iterable(a))
# print(b)
# ve={}
# with open(r'G:\my paper\预对齐嵌入向量\toy.vec') as f:
#     for a in f:
#         wav=a.split(' ')
#         a='de'
#         if a in wav:
#             d=[]
#             c=wav[1:]
#             # print(len(c))
#             for i in range(len(c)):
#                 d.append(float(c[i]))
#
#             ve[a]=[np.array(d)]
# print(ve)
# print(ve['de'])
# a={'ad':[31,4,24,46],'sv':[3,345,57,31]}
# b=a.keys()
# # print(list(b))
# initializer = np.random.uniform(-0.2, 0.2)
# s=[np.array([initializer]*300)]
# print(s)
# def count_nonzero(a, axis=None):
#     if axis==None:
#         print("1")
# # count_nonzero(2)
# from gensim.models import KeyedVectors
# w2v = KeyedVectors.load(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\model\w2v\w2v.model')
# # print(w2v)
# a=np.random.randn(300)
# b=np.random.randn(300)
# vectors = w2v.vectors
# # print(vectors.shape,type(vectors))
#
# d=np.concatenate((a.reshape(1, -1), vectors, b.reshape(1, -1)), axis=0)
# # print(d)
# print(type(d))
# # print(d.shape)
# with open(r'G:\my paper\预对齐嵌入向量\toy.vec') as f:
#     for a in f:
#         wav=a.split(' ')
#         a='de'
#         if a in wav:
#             b=[]
#             for i in range(len(wav[1:])):
#                 b.append(float(wav[1:][i]))
#             print(b)
#             print(np.array(b),type(np.array(b)))
# # a=[1,2,3]
# b=[3,0,1]
# # a=np.array(a)
# b=np.array(b)
# print(a)
# print(b)
# # if a[0]==b[2]:
# #     print("1")
# a=[1,2,3.4453,4,5,6,6]
# if 2.5 in a:
#     print("1")
# import pandas as pd
# import itertools
# biaoqian=pd.read_excel(r'D:\my paper aa\test\test_trans_labeled_token_split_1.xlsx',usecols=[2]).values.tolist()
# biaoqian=list(itertools.chain.from_iterable(biaoqian))
# print(biaoqian)
# print(sum((biaoqian)))
# print(2/3)
# data=pd.read_excel(r'D:\my paper aa\第一次整体结果整合\test_trans_labeled_token_split_1.xlsx')
# print(data)
# from sklearn.model_selection import GridSearchCV
# a=[[1,2,3],[0]]
# if 1 in a:
#     print("1")

# import pandas as pd
# data=pd.read_csv(r'D:\my paper aa\新随机合成训练集\reain2（人工标记的另外一部分未标记的训练集）\机器对齐结果\drive-download-20210107T144602Z-001\all_hebing_rengongjiancha.csv')
# for i in range(len(data)):
#     if data['zh_term'][i]=='zh_term':
#         data=data.drop(i)
# data=data.drop(['y_hat'],axis=1)
# data=data.drop(['trans'],axis=1)
# data.to_csv(r'D:\my paper aa\新随机合成训练集\reain2（人工标记的另外一部分未标记的训练集）\机器对齐结果\drive-download-20210107T144602Z-001\all_hebing_rengongjiancha1.csv',index=False)

# a=np.random.randn(300)
# print(a)
# c=np.random.randn(300)
# b=np.concatenate((a,c),axis=0)
# print(b)
# print(b.shape)
# import itertools
# print('create dict ...')
#     words = []
#     train = pd.read_excel(r'D:\my paper aa\train\train_trans_labeled_token1.xlsx')
#     for i in range(len(train)):
#         word1 = train['zh_term'][i].split()
#         words.append(word1)
#         word2 = train['es_term'][i].split()
#         words.append(word2)
#     test = pd.read_csv(r'D:\my paper aa\test\test_trans_labeled_token.csv')
#     for i in range(len(test)):
#         word1 = test['zh_term'][i].split()
#         words.append(word1)
#         word2 = test['es_term'][i].split()
#         words.append(word2)
# #     words = list(set(list(itertools.chain.from_iterable(words))))
# word_vector_dic = {}
# words=['de','la','en','el']
# with open(r'G:\my paper\预对齐嵌入向量\toy.vec',encoding='utf-8') as f:
#     for each_vec in f:
#         vector = each_vec.split(' ')
#         if vector[0] in words:
#             v = vector[1:]
#             v1 = []
#             for i in range(len(v)):
#                 v1.append(float(v[i]))
#             word_vector_dic[vector[0]] = np.array(v1)
# for i in range(len(words)):
#     if words[i] not in list(word_vector_dic.keys()):  ##没有找到对应的向量，则赋值[-0.2,0.2]之间的随机值
#         initializer = np.random.uniform(-0.2, 0.2)
#         word_vector_dic[words[i]] = [np.array([initializer] * 300)]
# f=word_vector_dic.values()
# h=[]
# for i in f:
#     h.append(np.array(i))
# h=np.array(h)
# # print(h,h.shape,type(h))
# tokens=[]
# import jieba
# a='我们 的 歌 asdd d_sd _ 221 呀'
# print(a.split())
# b=re.sub(r"[\w\d]+", " ", a, flags=re.ASCII)
# print(b)
# from gensim.models import KeyedVectors
# w2v = KeyedVectors.load(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_xgboost\model\w2v\w2v.model')
# # print(type(w2v))
# # # # file=open(r'd:\my paper aa\toy.txt')
# # # w2v=pd.DataFrame(list(w2v))
# # # w2v.to_csv(r'd:\my paper aa\toy.csv')
# a=w2v.index2word
# print(type(a))
# # w2v=pd.DataFrame(list(a))
# w2v.to_csv(r'd:\my paper aa\toy.csv')
# a=[1,2,3]
# a.insert(0, '<PAD>')
# a.add_word('UNK')
# print(a)
# a={1:'dwd',2:'wdq',3:'dswd'}
# # b=list(a.keys())
# c=list(a.values())
# d=dict(zip(c, range(len(c))))
# print(d)
# import numpy
# a=[1,2,3]
# # a=np.array(a)
# # print(type(a))
# # if isinstance(a, numpy.ndarray):
# #     print("d")
# # a={}
# # a[0]=np.array([1,2,3])
# # print(a)
# # b=[a.get(0)]
# # print(b[0],type(b[0]))
# class myself:
#     def __init__(self,r):
#         self.r=r
#     def __getitem__(self, index):
#         a=self.r[index]*2
#         b=2
#         c=3
#         d=self.pad2longest(b,c)
#         e=a*d
#         return e
#     @staticmethod
#     def pad2longest(data_ids, max_len):
#         d=data_ids*max_len
#         return d
# f=myself([1,2,3])
# print(f)
# import torch
# a=torch.rand(1,3)
# print(a)
# a = a > 0.5
# print(a)
# # print(a.shape)
# c=torch.squeeze(a)
# print(c)
# print(c.shape)
# c=list(c)
# c=pd.DataFrame(c)
# print(c)
# a=['sd','fe','efe']
# # print(a.index('sd'))
# a={'sds':np.array([1,2,3]),'da':np.array([2,4,6]),'as':np.array([5,7,8])}
# b=a.values()
# b=list(b)
# c=[]
# for i in range(len(b)):
#     c.append(np.array(b[i]))
# c=np.array(c)
# d=np.random.randn(3)
# d=d.reshape(1,-1)
# vectors = np.concatenate((d, c), axis=0)
# print(vectors,vectors.shape)
# f1=2
# if 2>1:
#     f1=3
# # print(f1)
# labels=[]
# data=pd.read_excel(r'd:\my paper aa\train\train_trans_labeled_token2.xlsx')
# for i in range(len(data)):
# #     labels.append(1) if data['biaoqian'][i] == 1 else labels.append(0)
# # print(labels[:63])
# # a=['ew','da','ssvzsf']
# # b=[1,2,3]
# # c=dict(zip(a, b))
# # print(c)
# import torch
# def metrics(labels, y_pred):
#     # labels and y_pred dtype is 8-bit integer
#     TP = ((y_pred == 1) & (labels == 1)).sum().float()
#     TN = ((y_pred == 0) & (labels == 0)).sum().float()
#     FN = ((y_pred == 1) & (labels == 0)).sum().float()
#     FP = ((y_pred == 0) & (labels == 1)).sum().float()
#     p = TP / (TP + FP).clamp(min=1e-8)
#     r = TP / (TP + FN).clamp(min=1e-8)
#     F1 = 2 * r * p / (r + p).clamp(min=1e-8)
#     acc = (TP + TN) / (TP + TN + FP + FN).clamp(min=1e-8)
#     return TP,TN,FN,FP,acc, p, r, F1
# a=torch.tensor([[0.212],[0.5464],[0.2313],[0.465],[0.6767]])
# b=torch.tensor([[1],[0],[0],[0],[1]])
# a= torch.squeeze(a, 1).to('cpu')
# b = torch.squeeze(b, 1).to('cpu')
# label = b > 0.5
# y_predict = a > 0.5
# TP,TN,FN,FP,acc, p, r, F1 = metrics(label, y_predict)
# print(TP,TN,FN,FP,acc, p, r, F1 )
# import torch
# a=torch.tensor([1,2,3,4,4.55,5.66])
# print(a,type(a))
# b=list(a)
# print(b,type(b))
# # c=[]
# for i in b:
#     c.append(list(i))
# print(c)
# list = a.detach().numpy().tolist()
# print(list,type(list))
# # print(a,type(a))
# # # a=[[1,2,3],[4,5,6]]
# # # for i in a:
# #     i=pd.DataFrame(i)
# #     i.to_csv(r'C:\Users\admin\Desktop\2.xlsx',mode='a',index=False,header=0)
# import torch
# a=torch.tensor([[0],[0],[0],[1],[0],[0],[0],[1],[0],[0],[0]])
# y = torch.squeeze(a, 1).float().to('cpu')
# print(y)
# pos = torch.eq(y ,1).float()
# neg = torch.eq(y ,0).float()
# num_pos = torch.sum(pos)
# num_neg = torch.sum(neg)
# # print(num_neg)
# # print(num_pos)
# num_total = num_pos + num_neg
# alpha_pos = num_neg / num_total
# alpha_neg = num_pos / num_total
# # print(alpha_neg)
# # print(alpha_pos)
# weights = alpha_pos * pos + alpha_neg * neg
# print(weights)
# import torch
# loss=[]
# a=torch.tensor(1.223)
# b=a.detach().numpy()
# loss.append(b)
# a=torch.tensor(3.564623)
# b=a.detach().numpy()
# loss.append(b)
# print(loss,type(loss))
# print(sum(loss)/len(loss))
# import math
# a=[[1,2,3],[4,5,6],[2,4,5]]
# col_totals = [ sum(x) for x in zip(*a) ]
# print(col_totals)
# # b=[col_totals[i]/len(a) for i in range(len(col_totals))]
# # print(b)
# import torch
# from torch.utils.data.sampler import WeightedRandomSampler
# from torch.utils.data import DataLoader
# numDataPoints = 1000
# data_dim = 5
# bs = 100
# # Create dummy data with class imbalance 9 to 1
# data = torch.FloatTensor(numDataPoints, data_dim) ##数据
# # print(data,data.shape)
# target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),np.ones(int(numDataPoints * 0.1), dtype=np.int32)))  ##label
# # print('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
# class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
# # print(class_sample_count)
# weight = 1. / class_sample_count
# print(weight)
# samples_weight = np.array([weight[t] for t in target])
# # print(samples_weight)  ##每个数据集的权重
# samples_weight = torch.from_numpy(samples_weight) #就是torch.from_numpy()方法把数组转换成tensor
# # print(samples_weight)
# samples_weight = samples_weight.double()
# # print(samples_weight)
# # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
# # target = torch.from_numpy(target).long()
# # train_dataset = torch.utils.data.TensorDataset(data, target)
# # train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, sampler=sampler)
# # for i, (data, target) in enumerate(train_loader):
# #     print( "batch index {}, 0/1: {}/{}".format(i,len(np.where(target.numpy() == 0)[0]),len(np.where(target.numpy() == 1)[0])))
# target=[1,0,0,1,1,0,0,1,0,1]
# print(np.unique(target))
# class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
# print(class_sample_count)
# weight = 1. / class_sample_count
# print(weight)
# data=pd.read_excel(r'C:\Users\admin\Desktop\1.xlsx',names=['zh_term','es_term','biaoqian'])
# for i in range(len(data)):
#     print(data['zh_term'][i].split())

# input1 = torch.randn(3, requires_grad=True)
# print(input1)
# y = torch.squeeze(torch.tensor([1,2,3,4]), 1).float().to('cpu')
# print(y,type(y))
# import torch
# import torch.nn as nn
# loss = nn.MarginRankingLoss(margin=0.6)
# score = torch.tensor([ 1.7831,  1.5558, -0.1122],requires_grad=True) ##y_hat
# print(score)
# y=torch.tensor([0,1,1])
# ya=list(y)
# input1=[]
# target=[]
# for i in range(len(ya)):
#     if ya[i]==0:
#         input1.append(1)
#         target.append(1)
#     else:
#         input1.append(0)
#         target.append(-1)
# input1=torch.tensor(input1)
# target=torch.tensor(target)
# output = loss(input1, score, target)
# print(output)
#
# # score1=list(score)
# # outputs=[]
# for i in range(len(ya)):
#     if ya[i]==1:
#         input1=torch.tensor([0])
#         input2=torch.tensor([score1[i]],requires_grad=True)
#         target=torch.tensor([-1])
#         output=loss(input1,input2,target)
#         outputs.append(output)
#         print("11",output)
#     if ya[i]==0:
#         input1=torch.tensor([1])
#         input2=torch.tensor([score1[i]],requires_grad=True)
#         target=torch.tensor([1])
#         output=loss(input1,input2,target)
#         outputs.append(output)
#         print("22", output)
# # print(outputs)
# # o=sum(outputs)/len(outputs)
# # print(o)
# # input1 = torch.randn(3, requires_grad=True)
# # input2 = torch.randn(3, requires_grad=True)
# target = torch.randn(3).sign()
# output = loss(input1, input2, target)
# print(output)
# import keras
# print(keras.__version__)
# # import tensorflow
# # print(tensorflow.__version__)
# # import matchzoo
# print(matchzoo.__version__)
# initializer = np.random.uniform(-0.2, 0.2)
# print([initializer]*3)
# a={'sd':[1,2,3],'sdsd':[4,5,6]}
# with open(r'c:\Users\admin\Desktop\q.txt',mode='w') as f:
#     for i in a:
#         f.write(i)
#         f.write(' ')
#         for j in range(len(a[i])):
#             f.write(str(a[i][j]))
#             f.write(' ')
#         f.write('\n')

# with open(r'C:\Users\admin\Desktop\q.txt') as f:
#     for i in a:
# matrix = np.empty((4, 5))
# initializer=lambda: np.random.uniform(-0.2, 0.2)
# for index in np.ndindex(*matrix.shape):
#     print(index)
#     matrix[index] = initializer()
# print(matrix)
# matrix = np.empty((4, 5))
# matrix[1] = [1,2,3,4,5]
# print(matrix)
# import csv
# data = pd.read_csv(r'C:\Users\admin\Desktop\q.txt',sep=" ", index_col=0,header=None,quoting=csv.QUOTE_NONE,usecols=[i for i in range(301)])
# print(data)
# a=np.array([[1],[2]])
# # a1=[]
# # for i in range(len(a)):
# #     a.
# word_vector_dnp.random.uniform(-0.2, 0.2)ic={}
# # words=['sd','dq']
# # np.random.seed(27)
# # initializer=lambda:
# a=[]
# with open(r'G:\my paper\pre_enb_vector\part_align_quchong.vec',encoding='utf-8') as f:
#     for line in f:
#         v=line.split(' ')
#         v1=[]
#         for i in range(1,len(v)-1):
#             v1.append(float(v[i]))
#         # print(len(v1))
#         a.append(v1)
# print(a)
# a='0.23123131'
# print(type(a))
# print(float(a))
import itertools
# words=['de','经济']
# word_vector_dic = {}
# with open(r'G:\my paper\pre_enb_vector\all_align_quchong.vec', encoding='utf-8') as f:  ##当使用了PART向量之后会出现问题，每行有一个'\n'，要去掉
#     for each_vec in f:
#         vector = each_vec.split(' ')
#         if vector[0] in words:
#             v = vector[1:]
#             v1 = []
#             for i in range(len(v)):
#                 v1.append(float(v[i]))
#             word_vector_dic[vector[0]] = np.array(v1)
# for i in range(len(words)):
#     if words[i] not in list(word_vector_dic.keys()):  ##没有找到对应的向量，则赋值[-0.2,0.2]之间的随机值
#         initializer = np.random.uniform(-0.2, 0.2)
#         word_vector_dic[words[i]] = np.array([initializer] * 300)
# # print(word_vector_dic)
# import collections
# lexicon = collections.defaultdict(set)
# lexicon[1].add(2)
# lexicon[1].add(3)
# print(lexicon)
# sc2 = np.zeros(2)
# print(sc2,type(sc2))
# result = np.random.randint(1, 13,(6,4))
# # print(result)
# # print(np.mean(result,axis=1))
# sc2=np.array([1,2,3])
# print(sc2)
# a=sc2[np.newaxis, :]
# print(a,a.shape)
# a=np.array([[1,2],[3,2],[1,4]])
# b=np.array([[4,2],[1,2],[2,4]])
# c=np.sum(a*b)
# print(c)
# a=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
# print(a)
# print(a.sum(1))

# # import pandas as pd
# data=pd.read_excel(r'C:\Users\admin\Desktop\联合国语料库2（完)1.xlsx')
# for i in range(len(data)):
#     if re.search('^del\s',data['term'][i])!=None or re.search('^ser \s',data['term'][i])!=None or re.search(r'\s ser$',data['term'][i])!=None or re.search('^tener \s',data['term'][i])!=None or re.search('\s tener$',data['term'][i])!=None or re.search('^de\s',data['term'][i])!=None or re.search('\sde$',data['term'][i])!=None or re.search('\sdel$',data['term'][i])!=None:
#         data['是否成词'][i]=0
#     else:
#         data['是否成词'][i]=1
# data.to_csv(r'C:\Users\admin\Desktop\联合国语料库改.csv',index=False,encoding='utf-8')
# import itertools
# data=pd.read_csv(r'G:\my paper\zh\zh_term_value_dc_dr_delzero.csv',usecols=[0]).values.tolist()
# word=list(itertools.chain.from_iterable(data))
# data1=pd.read_csv(r'C:\Users\admin\Desktop\22.csv')
# zh_term=[]
# es_term=[]
# biaoqian=[]
# for i in range(len(data1)):
#     if data1['zh_term'][i] not in word:
#         zh_term.append(data1['zh_term'][i])
#         es_term.append(data1['es_term'][i])
#         biaoqian.append(1)
# zh_term=pd.DataFrame(zh_term,columns=['zh_term'])
# es_term=pd.DataFrame(es_term,columns=['es_term'])
# biaoqian=pd.DataFrame(biaoqian,columns=['biaoqian'])
# data2=pd.concat([zh_term,es_term,biaoqian],axis=1)
# data2.to_csv(r'C:\Users\admin\Desktop\33.csv',index=False,encoding='utf-8')

# data=pd.read_excel(r'D:\my paper aa\新随机合成训练集\联合国语料库改删排筛.xlsx')
# # data1=pd.read_csv(r'D:\my paper aa\新随机合成训练集\train_trans_labeled0.csv',names=['zh_term','es_term','biaoqian'])
# import random
# index = random.sample(range(0,100),10)
# print(index)
# a=[['adad','koi','ada','efw','ada','hght']]
# import itertools
# b=list(set(list(itertools.chain.from_iterable(a))))
# print(b)
# import random
# a=[]
# for i in range(5):
#     index = random.sample(range(0,100),2)
#     a.append(index)
# print(a)
# data=pd.read_excel(r'C:\Users\admin\Desktop\test1.xlsx')
# data=data.sample(frac=1,random_state=None)
# data.to_csv(r'C:\Users\admin\Desktop\test4.csv',index=False)
# a=['wdd dwef fer','qwe ge']
# b=[]
# for i in a:
#     b.append(''.join(i.split()))
# print(b)
# data=pd.read_excel(r'C:\Users\admin\Desktop\11.xlsx',names=['zh_term','es_term','biaoqian'])
# zh_term=[]
# es_term=[]
# biaoqian=[]
# for i in range(len(data)):
#     a=data['zh_term'][i]
#     zh_term.append(''.join(a.split()))
#     es_term.append(data['es_term'][i])
#     biaoqian.append(data['biaoqian'][i])
# zh_term=pd.DataFrame(zh_term,columns=['zh_term'])
# es_term=pd.DataFrame(es_term,columns=['es_term'])
# biaoqian=pd.DataFrame(biaoqian,columns=['biaoqian'])
# data2=pd.concat([zh_term,es_term,biaoqian],axis=1)
# data2.to_csv(r'C:\Users\admin\Desktop\22.csv',index=False,encoding='utf-8')

# import pandas as pd
# import itertools
# data=pd.read_excel(r'C:\Users\admin\Desktop\源_分词_小写_去trans_预对齐_人工校正后2.xlsx',usecols=[0]).values.tolist()
# zh_word=list(set(list(itertools.chain.from_iterable(data))))
# print(len(zh_word))
# data3=pd.read_excel(r'C:\Users\admin\Desktop\源_分词_小写_去trans_预对齐_人工校正后2.xlsx')
# count=0
# for i in range(len(zh_word)):
#     group_data=data3[data3.zh_term==zh_word[i]]
#     group_data = group_data.reset_index()
#     group_data = group_data.drop(columns='index')
#     label=[]
#     for k in range(len(group_data)):
#         label.append(group_data['biaoqian'][k])
#     if 1 in label:
#         count+=1
#         group_data.to_csv(r'C:\Users\admin\Desktop\源_分词_小写_去trans_预对齐_人工校正后筛选2.csv',index=False,header=0,encoding='utf-8',mode='a')
# print(count)

# import pandas as pd
# import itertools
# data_train=pd.read_excel(r'D:\法语处理\训练集数据\训练集5000.xlsx',usecols=[0]).values.tolist()
# train=list(set(list(itertools.chain.from_iterable(data_train))))
# data_test=pd.read_excel(r'D:\法语处理\训练集数据\测试集625.xlsx',usecols=[0]).values.tolist()
# test=list(set(list(itertools.chain.from_iterable(data_test))))
# count=0
# for i in range(len(train)):
#     for j in range(len(test)):
#         if train[i]==test[j]:
#             print(test[j])
#             count+=1
# print(count)

a=['我们 的 歌']
c=[]
for i in a:
    b=i.strip().split()
    c.append(''.join(b))
print(c)