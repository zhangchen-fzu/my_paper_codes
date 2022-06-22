#西语术语度计算
import pandas as pd
import itertools
import os
import math
import re
data=pd.read_excel(r'G:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10_delchild1.xlsx',usecols=[0]).values.tolist()
data=list(itertools.chain.from_iterable(data))
term_value=[]
for i in range(len(data)):
    print(i)
    each_text_counts = []
    subdirs = os.walk(r'G:\my paper\es_huanyuan_addspace')   #文档集
    for d, s, fns in subdirs:
        for fn in fns:
            each_text_count=0
            with open(d + os.sep + fn, "r", encoding="utf-8") as p:
                text_content=p.read()
                if re.search(' '+data[i]+' ',text_content) !=None:
                    each_text_count+=text_content.count(' '+data[i]+' ')
            each_text_counts.append(each_text_count)
    # print(each_text_counts)
    count=0
    for i in each_text_counts:
        if i!=0:
            count+=1
    a=0
    for i in range(len(each_text_counts)):
        a+=(each_text_counts[i]-((sum(each_text_counts)*201)/(200*(count+1))))**2
    a=a+((sum(each_text_counts)/200)-((sum(each_text_counts)*201)/(200*(count+1))))**2
    b=(a/count)**(1/2)
    c=b*sum(each_text_counts)*(math.log((200/count)))
    term_value.append(c)
data1=pd.read_excel(r'G:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10_delchild1.xlsx',usecols=[0])
term_value=pd.DataFrame(term_value,columns=['term_value'])
data2=pd.concat([data1,term_value],axis=1)
data2=data2.sort_values(by=["term_value"],ascending=False)
data2.to_csv(r'G:\my paper\es\es_term_value.csv',index=False,encoding='utf-8')
print("术语度计算完毕！")
