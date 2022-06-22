##不太标准的计算方法！N个之中任意一个比对成功就算预测正确了,一个原词对应的翻译有一个对就算一个，有两个就算两个，所以会超过1
##为了是结果不超过1，改为不管原词对应的翻译找到了几个都算作一个！！参照 percision at topn!!
import pandas as pd
import itertools
import os


data=pd.read_csv(r'D:\my paper aa\第四次整体结果整合\all.csv')
word=pd.read_csv(r'D:\my paper aa\第四次整体结果整合\all.csv',usecols=[0]).values.tolist()
word=list(set(list(itertools.chain.from_iterable(word))))  #去重打乱
all_num=len(word)

def FindList1MaxNum(foo):
    max_values=[]
    max_value=max(foo)
    max_values.append(max_value)
    return max_values

def FindList2MaxNum(foo):
    max1,max2=None,None
    for num in foo:
        if max1 is None or max1<num:
            max1,num=num,max1
        if num is None:
            continue
        if max2 is None or num>max2:
            max2=num
    max_values=[]
    max_values.append(max1)
    max_values.append(max2)
    return max_values

def FindList3MaxNum(foo):
    max1, max2, max3 = None, None, None
    for num in foo:
        if max1 is None or max1 < num:
            max1, num = num, max1
        if num is None:
            continue
        if max2 is None or num > max2:
            max2, num = num, max2
        if num is None:
            continue
        if max3 is None or num > max3:
            max3 = num
    max_values=[]
    max_values.append(max1)
    max_values.append(max2)
    max_values.append(max3)
    return max_values

def FindList4MaxNum(foo):
    max1,max2,max3,max4=None,None,None,None
    for num in foo:
        if max1 is None or max1<num:
            max1,num=num,max1
        if num is None:
            continue
        if max2 is None or max2<num:
            max2,num=num,max2
        if num is None:
            continue
        if max3 is None or max3<num:
            max3,num=num,max3
        if num is None:
            continue
        if max4 is None or max4<num:
            max4=num
    max_values=[]
    max_values.append(max1)
    max_values.append(max2)
    max_values.append(max3)
    max_values.append(max4)
    return max_values

def FindList5MaxNum(foo):
    max1, max2, max3, max4,max5 = None, None, None, None,None
    for num in foo:
        if max1 is None or max1<num:
            max1,num=num,max1
        if num is None:
            continue
        if max2 is None or max2<num:
            max2,num=num,max2
        if num is None:
            continue
        if max3 is None or max3<num:
            max3,num=num,max3
        if num is None:
            continue
        if max4 is None or max4<num:
            max4,num=num,max4
        if num is None:
            continue
        if max5 is None or max5<num:
            max5=num
    max_values=[]
    max_values.append(max1)
    max_values.append(max2)
    max_values.append(max3)
    max_values.append(max4)
    max_values.append(max5)
    return max_values

num_precision=0
# choice_num=0
for i in range(len(word)):
    group_data=data[data.zh_term==word[i]]
    value=[]
    group_data = group_data.reset_index()
    group_data = group_data.drop(columns='index')
    for i in range(len(group_data)):
        value.append(group_data['y_hat'][i])
    max_value=FindList5MaxNum(value)
    for i in range(len(group_data)):
        if group_data['y_hat'][i] in max_value:
            # choice_num+=1
            if group_data['biaoqian'][i]==1:
                num_precision+=1
pre=num_precision/all_num
print("准确率：",pre)












