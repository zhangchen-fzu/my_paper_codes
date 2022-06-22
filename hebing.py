# import os
# import pandas as pd
# subdirs = os.walk(r'D:\my paper xiugai\验证集\分成多个xlsx')   # os.walk读取每个子目录下的文件
# for d, s, fns in subdirs:
#     for fn in fns:
#         if fn[-4:] == 'xlsx':
#             data=pd.read_excel(d + os.sep + fn)
#             data.to_csv(r'D:\my paper xiugai\验证集\分成多个xlsx\all_hebing.csv',mode='a',index=False)


# 人工检查标记
import pandas as pd
import itertools
data=pd.read_csv(r'D:\my paper xiugai\测试集\分成多个xksx\all_hebing_trans.csv')
word=pd.read_csv(r'D:\my paper xiugai\测试集\分成多个xksx\all_hebing_trans.csv',usecols=[0]).values.tolist()
word=list(set(list(itertools.chain.from_iterable(word))))  #去重打乱
# all_num=len(word)
for i in range(len(word)):
    group_data=data[data.zh_term==word[i]]
    group_data = group_data.reset_index()
    group_data = group_data.drop(columns='index')
    group_data=group_data.sort_values(by=['y_hat'],ascending=False)
    group_data.to_csv(r'D:\my paper xiugai\测试集\分成多个xksx\all_hebing_trans_jiancha.csv',mode='a',header=0,index=False)

#打散样本
# import pandas  as pd
# import itertools
# data=pd.read_excel(r'D:\my paper aa\train\train_trans_labeled_token2.xlsx')
# data=data.sample(frac=1,random_state=None)
# data.to_csv(r'D:\my paper aa\train\train_trans_labeled_token2_shuffle.csv',index=False)
#


##knrm_torch预测结果的合并
# import pandas as pd
# data1=pd.read_csv(r'D:\my paper aa\reain2\train_set2_zhtoken_bigtosmall.csv')
# data2=pd.read_csv(r'D:\my paper aa\reain2\机器对齐结果\drive-download-20210107T144602Z-001\all.csv')
# # data4=pd.read_csv(r'D:\my paper aa\train\train_trans_notlabeled.csv',names=['zh_term','es_term','trans','biaoqian'],usecols=[2])
# data3=pd.concat([data1,data2],axis=1)
# data3.to_csv(r'D:\my paper aa\reain2\机器对齐结果\drive-download-20210107T144602Z-001\all_hebing.csv',index=False,encoding='utf-8')


# 按中文打散数据
# import pandas as pd
# import itertools
# data=pd.read_excel(r'C:\Users\admin\Desktop\1.xlsx')
# word=pd.read_excel(r'C:\Users\admin\Desktop\1.xlsx',usecols=[0]).values.tolist()
# word=list(set(list(itertools.chain.from_iterable(word))))  #去重打乱
# all_num=len(word)
# for i in range(len(word)):
#     group_data=data[data.zh_term==word[i]]
#     group_data = group_data.reset_index()
#     group_data = group_data.drop(columns='index')
#     group_data = group_data.sample(frac=1, random_state=None)  ##内部打散
#     group_data.to_csv(r'C:\Users\admin\Desktop\2.csv',index=False,mode='a',header=0,encoding='utf-8')


##add方法合并数据
# import pandas as pd
# data=pd.read_csv(r'D:\my paper aa\新的7训练3测试\test30\test30percent.csv')
# data1=pd.read_csv(r'D:\my paper aa\新的7训练3测试\result\knrmnwemb\test_set_split1_result60.csv')
# data2=pd.concat([data,data1],axis=1)
# data2.to_csv(r'D:\my paper aa\新的7训练3测试\result\knrmnwemb\test_set_split1_result60_hebing.csv',index=False,encoding='utf-8')


##合并上trans
# import pandas as pd
# import itertools
# data=pd.read_csv(r'd:\my paper xiugai\验证集\val_token_bigtsmall.csv',encoding='utf-8',usecols=[2]).values.tolist()
# tran=list(itertools.chain.from_iterable(data))
# tran=pd.DataFrame(tran,columns=['tran'])
# data1=pd.read_csv(r'd:\my paper xiugai\验证集\分成多个xlsx\all_hebing.csv',encoding='utf-8')
# data2=pd.concat([data1,tran],axis=1)
# data2.to_csv(r'd:\my paper xiugai\验证集\分成多个xlsx\all_hebing_trans.csv',index=False,encoding='utf-8')