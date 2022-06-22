#es开头的序号删掉
# import os
# import re
# subdirs = os.walk(r'E:\my paper\es_add_biaodian')
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             with open(r'E:\my paper\es_del_num\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                     for eachline in p:
#                         word=eachline.split()
#                         words=[]
#                         for i in range(len(word)):
#                             if i==0:
#                                 # print(word[i])
#                                 if re.search("^\d+\.", word[0]) ==None and re.search("^[a-zA-Z]\)", word[0]) ==None and re.search("^[i]*\.",word[0])==None and re.search("^[I]*\.",word[0])==None and re.search("^IV\.",word[0])==None and re.search("^[a-zA-Z]\.", word[0]) ==None and re.search("^V\.",word[0])==None and re.search('^[i]*\)',word[0])==None:
#                                     words.append(word[0])
#                             else:
#                                 words.append(word[i])
#                         line=' '.join(words)
#                         if line:
#                             # print(line)
#                             f2.write(line+'\n')
# print("完成")

#一句话开头的字符串大写变小写
import os
import re
subdirs = os.walk(r'g:\my paper\背景语料库西语')
for d, s, fns in subdirs:
    for fn in fns:
        if fn[-3:] == 'txt':
            with open(r'G:\my paper\es_背景语料_big_tosmall\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
                with open(d + os.sep + fn, "r", encoding="utf-8") as p:
                    for eachline in p:
                        word=eachline.split()
                        words=[]
                        for i in range(len(word)):
                            if i==0:
                                words.append(word[i].lower())
                            else:
                                words.append(word[i])
                        line=' '.join(words)
                        f2.write(line+'\n')
print("完成")