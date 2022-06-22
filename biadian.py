#为每行后面添加标点
# import os
# subdirs = os.walk(r'E:\my paper\zh_text')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             with open(r'E:\my paper\zh_add_biaodian\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                     for eachline in p:
#                         eachline=eachline.strip('\n')
#                         eachline=eachline+'。'
#                         f2.write(eachline)
#                         f2.write('\n')
# print("标点添加完毕！")

# import os
# subdirs = os.walk(r'E:\my paper\es_text')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-3:] == 'txt':
#             with open(r'E:\my paper\es_add_biaodian\{}.txt'.format(fn[0:-4]), 'w', encoding='utf-8') as f2:
#                 with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                     for eachline in p:
#                         eachline=eachline.strip('\n')
#                         eachline=eachline+' .'  #注意加空格
#                         f2.write(eachline)
#                         f2.write('\n')
# print("标点添加完毕！")


