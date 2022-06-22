##参照人工构建的翻译词典 找到510万西语的汉语翻译
# import pandas as pd
# term=pd.read_csv(r'g:\my paper\candidate_term_quchong_val.csv',header=None, names = ['zh_term','es_term'])
# tran=pd.read_excel(r'C:\Users\admin\Desktop\翻译字典.xlsx')
#
# for i in range(len(tran)):  #3万
#     if i%10000==0:
#         print(i)
#     pos_samples = term[term.es_term == tran['term'][i]]
#     pos_samples['trans']=tran['tran'][i]
#     pos_samples.to_csv(r'D:\val_tran.csv',index=False, header=False, mode='a')

##加标签，并按中文排序
# import pandas as pd
# data=pd.read_csv(r'D:\val_tran.csv',names = ['zh_term','es_term','tran'])
# data=data.sort_values(by=['zh_term'],ascending=False)
# data.to_csv(r'D:\val_trans.csv',index=False)


##加0标签
# import pandas as pd
# data=pd.read_csv(r'D:\my paper xiugai\candidate_termpair_rank_quchong.csv',encoding='utf-8')
# label=[]
# for i in range(len(data)):
#     label.append(0)
# label=pd.DataFrame(label,columns=['label'])
# data1=pd.concat([data,label],axis=1)
# data1.to_csv(r'D:\my paper xiugai\candidate_termpair_rank_quchong_labels0.csv',index=False,encoding='utf-8')


# ##机器加标签1，如未识别出来人工判断
# import itertools
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
# term=pd.read_csv(r'd:\my paper xiugai\candidate_termpair_rank_quchong_labels0.csv',encoding='utf-8',usecols=[0]).values.tolist()
# term=list(itertools.chain.from_iterable(term))
# zh_terms=list(set(term))
# print(len(zh_terms))
#
# data=pd.read_csv(r'd:\my paper xiugai\candidate_termpair_rank_quchong_labels0.csv',encoding='utf-8',names=['zh_term','es_term','trans','label'])
# for i in range(len(zh_terms)):
#     if i%1000==0:
#         print(i)
#     group_daframe=data[data.zh_term==zh_terms[i]]  #dataframe类型
#     group_daframe=group_daframe.reset_index()
#     group_daframe = group_daframe.drop(columns='index')
#     label=[]
#     for i in range(len(group_daframe)):
#         if group_daframe['zh_term'][i]==group_daframe['trans'][i]:
#             group_daframe['label'][i]=1
#         label.append(group_daframe['label'][i])
#     if 1 in label:  #机器能够判断出来的
#         group_daframe.to_csv(r'd:\my paper xiugai\candidate_termpair_rank_quchong_labels0_labeled.csv',index=False, header=False, mode='a')
#     else:  #机器未能判断出来的
#         group_daframe.to_csv(r'd:\my paper xiugai\candidate_termpair_rank_quchong_labels0_notlabeled.csv',index=False, header=False, mode='a')
# print("机器过滤翻译完成！还需人工过滤notlabeled文件！")

