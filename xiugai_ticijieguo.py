# import pandas as pd
# data6=pd.read_excel(r'G:\my paper\es\final_result.xlsx',encoding='utf-8')
# word_fre_len_1_delchild=data6.sort_values(by=["freq"],ascending=False)
# word_fre_len_1_delchild1=word_fre_len_1_delchild[word_fre_len_1_delchild['freq']>1]
# word_fre_len_1_delchild1.to_csv(r'G:\my paper\es\without_stopwords_deldela_freq_freqbig1_deldela_dellenbig10_delchild.csv',encoding='utf-8',index=False)
# print("提词完毕！")
# import re
# a='sada cdsf fsd'
# b='sdjwe sada cdsf fsd'
# print(b.count(a))
# import re
# each_text_counts=[]
# each_text_count=0
# with open(r'G:\my paper\es_huanyuan\2012_td_b_c_i_24..txt', "r", encoding="utf-8") as p:
#     text_content = p.read()
#     if re.search( 'ser la', text_content) != None:
#         each_text_count += text_content.count( ' '+'ser la')
#     if re.search('ser la' + ' ', text_content) != None:
#         each_text_count += text_content.count('ser la'+' ')
#     if re.search(' ' + 'ser la' + ' ', text_content) != None:
#         each_text_count += text_content.count(' ' + 'ser la'+' ')
# each_text_counts.append(each_text_count)
# print(each_text_counts)


