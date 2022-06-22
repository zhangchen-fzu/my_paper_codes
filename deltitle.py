#es删除题目
import os
subdirs = os.walk(r'E:\my paper\es_big_tosmall')   # os.walk读取每个子目录下的文件
for d, s, fns in subdirs:
    for fn in fns:
        if fn[-3:] == 'txt':
            with open(r'E:\my paper\es_text_set\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
                with open(d + os.sep + fn, "r", encoding="utf-8") as p:
                    text=[]
                    for eachline in p:
                        text.append(eachline)
                    for i in range(len(text)):
                        if i!=1 and i!=3 and i!=4 and i!=5:
                            f2.write(text[i])
print("无法对齐的题目已删除完毕！")

#zh删除题目
# import os
# subdirs = os.walk(r'E:\my paper\zh_add_biaodian')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             with open(r'E:\my paper\zh_text_set\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                     text=[]
#                     for eachline in p:
#                         text.append(eachline)
#                     for i in range(len(text)):
#                         if i!=1 and i!=3 and i!=4 and i!=5:
#                             f2.write(text[i])
# print("无法对齐的题目已删除完毕！")