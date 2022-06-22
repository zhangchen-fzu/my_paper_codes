import treetaggerwrapper as tter
import pandas as pd
import re
import os
from collections import Counter
import numpy as np

#加载停用词、停用词性表
tagger = tter.TreeTagger(TAGLANG='es')
word_dict = {}
stop_words = set([line.strip() for line in open(r'E:\my paper\esstoplist.txt',encoding='utf-8').readlines()])
pos_deleted = set(['ALFP', 'ALFS', 'BACKSLASH', 'CC', 'CCAD', 'CCNEG', 'CM', 'CODE','COLON', 'CQUE', 'CSUBF', 'CSUBI', 'CSUBX', 'DASH', 'DM', 'DOTS', 'FO', 'INT', 'ITJN', 'LP', 'ORD', 'PERCT', 'PPC', 'PPO', 'PPX',  'QT', 'QU', 'REL', 'RP', 'SE', 'SEMICOLON', 'SLASH', 'SYM', 'UMMX', 'CARD', 'FS'])  # 根据停用词性表，删除无用原子词

#词性标注+词形还原
words_list_pos = []
words_list_huanyuan = []
with open(r'E:\my paper\es_text_set\es_alltext.txt', 'r', encoding='utf-8') as f2:
    with open(r'E:\my paper\es\without_stopwords.txt', 'w', encoding='utf-8') as f1:
        with open(r'E:\my paper\es\with_stopword_huanyuan.txt', 'w', encoding='utf-8') as f3:
            for each_line in f2:
                tags = tagger.tag_text(each_line)
                tags2 = tter.make_tags(tags, allow_extra=True)
                words_list1_huanyuan = []
                words_list1_pos = []
                for d in tags2:
                    # print(d)
                    if d[2] == '<unknown>' or d[2] == '@card@' or d[2] == '@ord@':
                        word_lemma = d[0]
                        f3.write(word_lemma + '/' + d[1] + ' ')
                        words_list1_huanyuan.append(word_lemma)
                        words_list1_pos.append(d[1])
                    else:
                        if d[0] in ["El",'Los']:
                            word_lemma = 'El'
                            f3.write(word_lemma+'/'+d[1]+' ')
                            words_list1_huanyuan.append(word_lemma)
                            words_list1_pos.append(d[1])
                        else:
                            if d[0] in ["el","los"]:
                                word_lemma = 'el'
                                f3.write(word_lemma+'/'+d[1]+' ')
                                words_list1_huanyuan.append(word_lemma)
                                words_list1_pos.append(d[1])
                            else:
                                if  d[0] in ["las","la"]:
                                    word_lemma='la'
                                    f3.write(word_lemma+'/'+d[1]+' ')
                                    words_list1_huanyuan.append(word_lemma)
                                    words_list1_pos.append(d[1])
                                else:
                                    if d[0] in ["Las","La"]:
                                        word_lemma='La'
                                        f3.write(word_lemma+'/'+d[1]+' ')
                                        words_list1_huanyuan.append(word_lemma)
                                        words_list1_pos.append(d[1])
                                    else:
                                        word_lemma = d[2]
                                        f3.write(word_lemma + '/' + d[1] + ' ')
                                        words_list1_huanyuan.append(word_lemma)
                                        words_list1_pos.append(d[1])
                f3.write('\n')
                words_list_without_stopwords = []
                df = {"term": words_list1_huanyuan, "pos": words_list1_pos}
                words_pos_DF = pd.DataFrame(df)
                for i in range(0, len(words_pos_DF)):
                    if words_pos_DF['term'][i] in stop_words or words_pos_DF['pos'][i] in pos_deleted:
                        words_list_without_stopwords.append('\n')
                    else:
                        words_list_without_stopwords.append(words_pos_DF['term'][i])
                f1.write(' '.join(words_list_without_stopwords))
                f1.write('\n')
                words_list_huanyuan.append(words_list1_huanyuan)
                words_list_pos.append(words_list1_pos)

# # #在去停用词之后的串中删除de、la等开头和结尾的,提前去掉，为了避免原子词步长法提取出更多的以他们开头和结尾的词语
# with open(r'E:\my paper\es\without_stopwords.txt','r', encoding='utf-8') as f1:
#     with open(r'E:\my paper\es\without_stopwords_deldela.txt','w', encoding='utf-8') as fn:
#         for each_line in f1:
#             each_line = re.sub("^de\s", '', each_line)
#             each_line = re.sub("^du\s", '', each_line)
#             each_line = re.sub("^le\s", '', each_line)
#             each_line = re.sub("^la\s", '', each_line)
#             each_line = re.sub("^del\s", '', each_line)
#             each_line = re.sub("\sla de$", '',each_line)
#             each_line = re.sub("\sde la$", '', each_line)
#             each_line = re.sub("\sel de$", '', each_line)
#             each_line = re.sub("\sde el$", '', each_line)
#             each_line = re.sub("\sla el$", '', each_line)
#             each_line = re.sub("\sel la$", '', each_line)
#             each_line = re.sub("\sde\s$", '', each_line)
#             each_line = re.sub("\sdu\s$", '', each_line)
#             each_line = re.sub("\sle\s$", '', each_line)
#             each_line = re.sub("\sla\s$", '', each_line)
#             each_line = re.sub("\sdel\s$", '', each_line)
#             fn.write(each_line)
#             fn.write('\n')

#原子词步长法
# def aswExtract(path):
#     with open(path, encoding='utf-8') as file:
#         awsList = []
#         for line in file:
#             awsList.append(line.split())
#     return awsList
# def getChild(listA, b, argv):
#     a = ' '
#     ChildList = []
#     for i in range(0, b - argv + 1):
#         ChildList.append(a.join(listA[i:i + argv]))
#     return ChildList
# awsList = aswExtract(r'E:\my paper\es\without_stopwords_deldela.txt')
# result = []
# for aws in awsList:
#     temp = []
#     b = len(aws)
#     for argv in range(1, b + 1):
#         temp.append(getChild(aws, b, argv))
#     result.append([j for k in temp for j in k])
# # 统计词频
# wordDict = {}
# for x in result:
#     d = Counter(x)
#     for key, value in d.items():
#         wordDict[key] = wordDict.get(key, 0) + value
# f = pd.DataFrame(pd.Series(wordDict), columns=['freq'])
# f = f.reset_index().rename(columns={'index': 'term'})
# output = f.sort_values(by=['freq'], ascending=False)
# output.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq.csv', index=False, encoding='utf-8')
# # 找到频数大于1的,到此词语提取结束，但是还增加了对西语额外的处理：特殊字符开头以及子串的删除
# data=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq.csv',encoding='utf-8')
# data1=data[data['freq']>1]
# data1.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1.csv', index=False, encoding='utf-8')

# # #删除之前因为词串中间有de、la等而出现的以他们开头的词语的前后缀
# data4=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1.csv',  encoding='utf-8')
# for i in range(0,len(data4)):
#     print(i)
#     data4['term'][i] = re.sub("^de\s", '', data4['term'][i])
#     data4['term'][i] = re.sub("^del\s", '', data4['term'][i])
#     data4['term'][i] = re.sub("^de el\s", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sla de$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sde la$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sel de$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sde el$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sla el$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sel la$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sde$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sel$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sla$", '', data4['term'][i])
#     data4['term'][i] = re.sub("\sdel$", '', data4['term'][i])
#     data4['term'][i] = re.sub("^la$", '', data4['term'][i])
#     data4['term'][i] = re.sub("^de$", '', data4['term'][i])
#     data4['term'][i] = re.sub("^el$", '', data4['term'][i])
#     data4['term'][i] = re.sub("^del$", '', data4['term'][i])
# length_dict = {}
# for i in range(len(data4)):
#     a = data4['term'][i]
#     b = len(a.split())
#     length_dict.setdefault(b,[]).append(a)
# length_dict = dict(sorted(length_dict.items(),key=lambda item:item[0], reverse=True))
# data=list(length_dict.values())
# term_list = []
# for j in data:
#     term_list.extend(j) #词语长度从长到短排列的词语列表
# df2 = pd.DataFrame(term_list)
# df2.rename(columns={0:"term"},inplace=True)
# df=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq.csv',encoding='utf-8')
# df3=pd.merge(df2,df,on='term',how='inner')
# df3=df3.drop_duplicates()
# df3=df3.reset_index()
# df3=df3.drop(columns='index')
# df3.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela.csv',encoding='utf-8',index=False)

#去子串,长度大于10的删除
# data3=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela.csv',encoding='utf-8')
# lens=[]
# for i in range(len(data3)):
#     lens.append(len(data3['term'][i].split()))
# lens=pd.DataFrame(lens,columns=['lens'])
# data4=pd.concat([data3,lens],axis=1)
# data5=data4[data4['lens']<11]
# data6=data5.sort_values(by=['lens'], ascending=False)
# data6.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10.csv',encoding='utf-8',index=False)
#
# data6=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10.csv',encoding='utf-8')
# for i in range(len(data6)):
#     print(i)
#     length=data6['lens'][i]
#     for j in range(0,i):
#         if length<data6['lens'][j]:
#             if re.search("\s+" + data6['term'][i] + "\s", data6['term'][j]) != None or re.search("\s+" + data6['term'][i], data6['term'][j]) != None or re.search(data6['term'][i]+"\s",data6['term'][j])!=None:
#                 data6['freq'][i] = data6['freq'][i] - data6['freq'][j]
# word_fre_len_1_delchild=data6.sort_values(by=["freq"],ascending=False)
# word_fre_len_1_delchild1=word_fre_len_1_delchild[word_fre_len_1_delchild['freq']>1]
# word_fre_len_1_delchild1.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_lens_delchild.csv',encoding='utf-8',index=False)
# print("提词完毕！")

# data6=pd.read_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10.csv',encoding='utf-8')
# for i in range(0,len(data6)):
#     print(i)
#     for j in range(0,i):
#         if re.search("\s+"+data6['term'][i]+"\s",data6['term'][j])!=None or re.search("\s+"+data6['term'][i],data6['term'][j])!=None \
#                 or re.search(data6['term'][i]+"\s",data6['term'][j])!=None:
#             data6['freq'][i] = data6['freq'][i] - data6['freq'][j]
# word_fre_len_1_delchild=data6.sort_values(by=["freq"],ascending=False)
# word_fre_len_1_delchild1=word_fre_len_1_delchild[word_fre_len_1_delchild['freq']>1]
# word_fre_len_1_delchild1.to_csv(r'E:\my paper\es\without_stopwords_deldela_freq_freqbig1_lens_delchild.csv',encoding='utf-8',index=False)
# print("提词完毕！")
