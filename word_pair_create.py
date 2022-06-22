# #合并两个txt文件,其实在计算笛卡尔积的时候无需合并文件
# es_data=[]
# with open(r'g:\my paper\es_alltext(对alltext的还原).txt','r',encoding='utf-8') as es_file:
#     for eachline in es_file:
#         eachline=eachline.strip()
#         es_data.append(' '+eachline)
# print("1")
# zh_data=[]
# with open(r'g:\my paper\zh_alltext（也未做过任何处理，与上边的text是对齐的，与text1未对齐）.txt','r',encoding='utf-8') as zh_file:
#     for eachline in zh_file:
#         eachline=eachline.strip()
#         zh_data.append(eachline)
# print("2")
# with open(r'g:\my paper\zh_es_alltext(中西合并结果).txt','w',encoding='utf-8') as f:
#     for i in range(len(zh_data)):
#         f.write(zh_data[i]+'\n')
#         f.write(es_data[i]+'\n')
# print("合并完成！")

##笛卡尔积
# import itertools
# import pandas as pd
#
# zh_term=[]
# data=pd.read_csv(r'g:\my paper\zh\zh_term_value_dc_dr_delzero_drdcchengji.csv',encoding='utf-8')
# for i in range(len(data)):
#     zh_term.append(data['word'][i])
# zh_all_line_term=[]  #有多少句话就有多少个[]存在，二维列表
# with open(r'g:\my paper\zh_alltext（也未做过任何处理，与上边的text是对齐的，与text1未对齐）.txt','r',encoding='utf-8') as zh_file:
#     for eachline in zh_file:
#         zh_each_line_term=[]
#         for j in range(len(zh_term)):
#             if str(zh_term[j]) in str(eachline):
#                 zh_each_line_term.append(zh_term[j])
#         zh_all_line_term.append(zh_each_line_term)
# print("1")
#
# es_term=[]
# data=pd.read_csv(r'g:\my paper\es\es_term_value_dc_dr_delzero_drdcchengji.csv',encoding='utf-8')
# for i in range(len(data)):
#     es_term.append(data['term'][i])
# es_all_line_term=[]  #有多少句话就有多少个[]存在
# with open(r'g:\my paper\es_alltext(对alltext的还原).txt','r',encoding='utf-8') as es_file:
#     for eachline in es_file:
#         es_each_line_term=[]
#         for j in range(len(es_term)):
#             if ' '+str(es_term[j])+' ' in str(eachline):
#                 es_each_line_term.append(es_term[j])
#         es_all_line_term.append(es_each_line_term)
# print("2")
#
# all_dicard=[]  #三维列表
# for i in range(len(zh_all_line_term)):
#     if zh_all_line_term[i]!=[] and es_all_line_term[i]!=[]:
#         dicard=[]
#         for j in zh_all_line_term[i]:
#             for k in es_all_line_term[i]:
#                 dicard.append([j,k])
#         all_dicard.append(dicard)
# all_dicard=list(itertools.chain.from_iterable(all_dicard))  #二维列表
# zh=[]
# es=[]
# for i in range(len(all_dicard)):
#     zh.append(all_dicard[i][0])
#     es.append(all_dicard[i][1])
# zh=pd.DataFrame(zh,columns=['zh_term'])
# es=pd.DataFrame(es,columns=['es_term'])
# result=pd.concat([zh,es],axis=1)
# result.to_csv(r'g:\my paper\candidate_term.csv',encoding='utf-8',index=False)
# print("笛卡尔积计算候选术语完成！")
#
##去重
# import pandas as pd
# term_pair_list=[]
# data=pd.read_csv(r'g:\my paper\candidate_term.csv',encoding='utf-8')
# print(len(data))
# data.drop_duplicates(inplace=True)
# print(len(data))
# data=data.sort_values(by=['zh_term'],ascending=False)
# data.to_csv(r'g:\my paper\candidate_term_quchong.csv',index=False,encoding='utf-8')
# print("完成！")

##预对齐向量繁化简
# from langconv import  *
# import numpy as np
# import io
# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     x = np.zeros([n, d])
#     words = []     #存放词语
#     for i, line in enumerate(fin):
#         tokens = line.rstrip().split(' ')
#         words.append(tokens[0])
#         v = np.array(tokens[1:], dtype=float)
#         x[i, :] = v     #存放向量
#     return words, x
#
# def save_vectors(fname, x, words): #位置 向量 词语
#     n, d = x.shape
#     fout = io.open(fname, 'w', encoding='utf-8')
#     fout.write(u"%d %d\n" % (n, d))
#     for i in range(n):
#         fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
#     fout.close()
#
# word,vector=load_vectors(r'G:\my paper\预对齐嵌入向量\wiki.zh.align.vec')
# word_jians=[]
# for i in range(len(word)):
#     word_jian= Converter('zh-hans').convert(word[i])
#     word_jians.append(word_jian)
#
# save_vectors(r'G:\my paper\预对齐嵌入向量\wiki.zh_jian.align.vec',vector,word_jians)
# print("预对齐向量繁化简完成！")