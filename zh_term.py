#中文术语度计算
import pandas as pd
import itertools
import os
import math
data=pd.read_excel(r'G:\my paper\zh\without_stopwords_wordlist_atomicfreq_lenrank_frebig1_delchild1.xlsx',usecols=[0]).values.tolist()
data=list(itertools.chain.from_iterable(data))
term_value=[]
for i in range(len(data)):
    print(i)
    count = 0
    each_text_count = []
    subdirs = os.walk(r'G:\my paper\zh_text_set')   #文档集
    for d, s, fns in subdirs:
        for fn in fns:
            if fn[-3:] == 'txt':
                    with open(d + os.sep + fn, "r", encoding="utf-8") as p:
                        text_content=p.read()
                        if data[i] in text_content:
                            each_text_count.append(text_content.count(data[i]))
                            count+=1
    a=0
    for i in range(len(each_text_count)):
        a+=(each_text_count[i]-((sum(each_text_count)*201)/(200*(count+1))))**2
    a=a+((sum(each_text_count)/200)-((sum(each_text_count)*201)/(200*(count+1))))**2
    b=(a/count)**(1/2)
    c=b*sum(each_text_count)*(math.log((200/count)))
    term_value.append(c)
data1=pd.read_excel(r'G:\my paper\zh\without_stopwords_wordlist_atomicfreq_lenrank_frebig1_delchild1.xlsx')
term_value=pd.DataFrame(term_value,columns=['term_value'])
data2=pd.concat([data1,term_value],axis=1)
data2=data2.sort_values(by=["term_value"],ascending=False)
data2.to_csv(r'G:\my paper\zh\zh_term_value.csv',index=False,encoding='utf-8')
print("术语度计算完毕！")