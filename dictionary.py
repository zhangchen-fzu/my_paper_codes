zh_en_dics=[]
with open(r'G:\my paper\字典（暂无用处，其中的中西是经过英语合并出来的）\zh-en.txt','r',encoding='utf-8') as zh_en_dic:
    for each_line1 in zh_en_dic:
        zh_en_dics.append(each_line1.split())

en_fr_dics=[]
with open(r'G:\my paper\字典（暂无用处，其中的中西是经过英语合并出来的）\en-fr.txt','r',encoding='utf-8') as en_es_dic:
    for each_line2 in en_es_dic:
        en_fr_dics.append(each_line2.split())

zh_fr_dics=[]
for i in range(len(zh_en_dics)):
    print(i)
    for j in range(len(en_fr_dics)):
        if zh_en_dics[i][1]==en_fr_dics[j][0]:
            zh_fr_dics.append([zh_en_dics[i][0],en_fr_dics[j][1]])


with open(r'G:\my paper\字典（暂无用处，其中的中西是经过英语合并出来的）\zh-fr.txt','w',encoding='utf-8') as w:
    for k in range(len(zh_fr_dics)):
        w.write(' '.join(zh_fr_dics[k]))
        w.write('\n')
print("词典构建完成！")
