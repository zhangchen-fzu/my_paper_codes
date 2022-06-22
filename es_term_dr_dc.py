import pandas as pd
import itertools
import os
import math
###dc
# data=pd.read_excel(r'g:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10_delchild1.xlsx',usecols=[0]).values.tolist()
# data=list(itertools.chain.from_iterable(data))
# term_value=[]  #dc
# for i in range(len(data)):
#     print(i)
#     count = 0  #df
#     each_text_count = []  #tf
#     subdirs = os.walk(r'g:\my paper\es_huanyuan_addspace')   #文档集
#     for d, s, fns in subdirs:
#         for fn in fns:
#             if fn[-3:] == 'txt':
#                     with open(d + os.sep + fn, "r", encoding="utf-8") as p:
#                         text_content=p.read()
#                         if ' '+ data[i]+' ' in text_content:
#                             each_text_count.append(text_content.count(' '+data[i]+' '))
#                             count+=1
#     a=0
#     for i in range(len(each_text_count)):
#         b=each_text_count[i]/sum(each_text_count)
#         c=b*math.log(1/b)
#         a+=c
#     term_value.append(a)
# data1=pd.read_excel(r'g:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10_delchild1.xlsx')
# term_value=pd.DataFrame(term_value,columns=['dc'])
# data2=pd.concat([data1,term_value],axis=1)
# data2.to_csv(r'G:\my paper\es\es_term_value_dc.csv',index=False,encoding='utf-8')
# print("术语度计算完毕！")


###dr
# data=pd.read_csv(r'g:\my paper\es\es_term_value_dc.csv',usecols=[0]).values.tolist()
# data=list(itertools.chain.from_iterable(data))
# def freq(data,path):
#     with open(path, "r", encoding="utf-8") as p:
#         text = p.read()
#         all_tfs = []  #存放每个词语的词频
#         for i in range(len(data)):
#             if ' '+data[i]+' ' in text:
#                 all_tfs.append(text.count(' '+data[i]+' '))
#             else:
#                 all_tfs.append(0)
#     return all_tfs
# all_tfs_domain=freq(data,r'g:\my paper\es_alltext1.txt')  #说明：es_alltext.txt是为了提词而形成的并且和中文的已对齐每句前未加空格，es_alltext1.txt是为了计算词频而形成的，每句前后都加了空格
# all_tfs_notdomain=freq(data,r'g:\my paper\es_背景语料_huanyuan_addspace\all_背景语料_text.txt')  #和上面的长度应保持一致为data的长度
# drs=[]
# for i in range(len(all_tfs_domain)):
#     dr=(math.log((all_tfs_domain[i]/200)/((all_tfs_notdomain[i]+1e-8)/280)))*(math.log(all_tfs_domain[i]))
#     drs.append(dr)
# data1=pd.read_csv(r'g:\my paper\es\es_term_value_dc.csv')
# drs=pd.DataFrame(drs,columns=['dr'])
# data2=pd.concat([data1,drs],axis=1)
# data2.to_csv(r'g:\my paper\es\es_term_value_dc_dr.csv',index=False,encoding='utf-8')
# print("术语度计算完毕！")

###筛选
# data=pd.read_csv(r'g:\my paper\es\es_term_value_dc_dr.csv',encoding='utf-8')
# data1=data[data['dc']>0]
# data2=data1[data1['dr']>=0]
# data2.to_csv(r'g:\my paper\es\es_term_value_dc_dr_delzero.csv',index=False,encoding='utf-8')
# print("完成！")

###dc和dr合并排序
data=pd.read_csv(r'g:\my paper\es\es_term_value_dc_dr_delzero.csv',encoding='utf-8')
dr_dc=[]
for i in range(len(data)):
    a=(0.95*data['dr'][i])+(0.05*data['dc'][i])
    dr_dc.append(a)
dr_dc=pd.DataFrame(dr_dc,columns=['0.95dr_0.05dc'])
data2=pd.concat([data,dr_dc],axis=1)
data2=data2.sort_values(by=["0.95dr_0.05dc"],ascending=False)
data2.to_csv(r'g:\my paper\es\es_term_value_dc_dr_delzero_drdcchengji.csv',index=False,encoding='utf-8')
print("术语度计算完毕！")