import treetaggerwrapper as tter
# import pandas as pd
# import re
# import os
# from collections import Counter
# import numpy as np
import os

# 加载停用词、停用词性表
# tagger = tter.TreeTagger(TAGLANG='es')
# subdirs = os.walk(r'g:\my paper\es_背景语料_big_tosmall')   # os.walk读取每个子目录下的文件
# for a, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             with open(r'G:\my paper\es_背景语料_huanyuan\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(a + os.sep + fn, "r", encoding="utf-8") as p:
#                     for eachline in p:
#                         eachline=eachline.strip()
#                         words_list1_huanyuan = []
#                         tags = tagger.tag_text(eachline)
#                         tags2 = tter.make_tags(tags, allow_extra=True)
#                         for d in tags2:
#                             if len(d)==3:   #=============粗略的估算时才可加，否则会有丢失的现象==========
#                                 if d[2] == '<unknown>' or d[2] == '@card@' or d[2] == '@ord@':
#                                     word_lemma = d[0]
#                                     words_list1_huanyuan.append(word_lemma)
#                                 else:
#                                     if d[0] in ["El", 'Los']:
#                                         word_lemma = 'El'
#                                         words_list1_huanyuan.append(word_lemma)
#                                     else:
#                                         if d[0] in ["el", "los"]:
#                                             word_lemma = 'el'
#                                             words_list1_huanyuan.append(word_lemma)
#                                         else:
#                                             if d[0] in ["las", "la"]:
#                                                 word_lemma = 'la'
#                                                 words_list1_huanyuan.append(word_lemma)
#                                             else:
#                                                 if d[0] in ["Las", "La"]:
#                                                     word_lemma = 'La'
#                                                     words_list1_huanyuan.append(word_lemma)
#                                                 else:
#                                                     word_lemma = d[2]
#                                                     words_list1_huanyuan.append(word_lemma)
#                         huanyuan_result=' '.join(words_list1_huanyuan)
#                         f2.write(huanyuan_result+'\n')
# print("文本还原完成！")
#
#

#首尾加空格
# import os
# subdirs = os.walk(r'G:\my paper\es_背景语料_huanyuan')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         with open(r'G:\my paper\es_huanyuan_addspace\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#             with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                 for eachline in p:
#                     eachline=eachline.strip('\n')
#                     eachline=' '+eachline+' '
#                     f2.write(eachline+'\n')
#
# print("完成！")



#搜索
# import os
# subdirs = os.walk(r'G:\my paper\es_huanyuan_addspace')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         # with open(r'G:\my paper\es_huanyuan_addspace\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#             with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                 a=p.read()
#                 if 'OFC' in a:
#                     print(fn)


#由于在提词时没有保留还原的资料 现进行还原处理 输入是提词输入的总文本text 输出是还原之后的总文本并在每句的开头加了空格 方便搜索词语 始终与西语对齐
# tagger = tter.TreeTagger(TAGLANG='es')
# with open(r'g:\my paper\es_alltext(对alltext的还原).txt', 'w', encoding='utf-8') as f2:
#     with open(r'g:\my paper\es_alltext（未做过任何处理，是提词用的，所有的包活还原之类的步骤都是在提词的时候进行的）.txt',encoding='utf-8') as esfile:
#         for eachline in esfile:
#             eachline = eachline.strip()
#             tags = tagger.tag_text(eachline)
#             tags2 = tter.make_tags(tags, allow_extra=True)
#             words_list1_huanyuan = []
#             for d in tags2:
#                 if d[2] == '<unknown>' or d[2] == '@card@' or d[2] == '@ord@':
#                     word_lemma = d[0]
#                     words_list1_huanyuan.append(word_lemma)
#                 else:
#                     if d[0] in ["El", 'Los']:
#                         word_lemma = 'El'
#                         words_list1_huanyuan.append(word_lemma)
#                     else:
#                         if d[0] in ["el", "los"]:
#                             word_lemma = 'el'
#                             words_list1_huanyuan.append(word_lemma)
#                         else:
#                             if d[0] in ["las", "la"]:
#                                 word_lemma = 'la'
#                                 words_list1_huanyuan.append(word_lemma)
#                             else:
#                                 if d[0] in ["Las", "La"]:
#                                     word_lemma = 'La'
#                                     words_list1_huanyuan.append(word_lemma)
#                                 else:
#                                     word_lemma = d[2]
#                                     words_list1_huanyuan.append(word_lemma)
#             huanyuan_result = ' '.join(words_list1_huanyuan)
#             f2.write(' '+huanyuan_result+'\n')
# print("文本还原完成！")