# ###在正式处理之前需对中文进行NLPIR分词，对西语进行大变小转换并去重！！！！！！！！！！
import torch
import numpy as np
import pandas as pd
import matchzoo as mz
#
# #定义任务类型和评价指标
print('task define ...')
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=10))  #任务类型为排序任务
ranking_task.metrics = [mz.metrics.MeanAveragePrecision()]  #最终的评价指标为MAP
# #
#加载自己的嵌入向量
print("embedding load ...")
emb = mz.embedding.load_from_file(r'g:\my paper\预对齐嵌入向量\all_align_quchong.vec',mode='fasttext')
# print("embedding loaded as `glove_embedding`")
#
# #加载数据，转换数据类型，划分训练集和测试集，定义数据
print('data loading ...')
train = pd.read_excel(r'c:\Users\admin\Desktop\train_file.xlsx',encoding='utf-8')
train.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
vald=pd.read_excel(r'c:\Users\admin\Desktop\vald_file.xlsx',encoding='utf-8')
vald.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
test=pd.read_excel(r'c:\Users\admin\Desktop\test_file.xlsx',encoding='utf-8')
test.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
train_pack = mz.pack(train)
vald_pack = mz.pack(vald)
test_pack=mz.pack(test)
# print(train_pack.frame())
# print(vald_pack.frame())
# print(test_pack.frame())

#
# #数据预处理
print('data preproces ...')
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=10, remove_stop_words=False)
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(vald_pack)
test_processed = preprocessor.transform(test_pack)
# # print(preprocessor.context)
# # print(preprocessor.context['vocab_unit'].state['term_index'])
# # #
# # # # # ##模型参数赋值
print('model parm define ...')
bin_size = 5   #直方图的维度
model = mz.models.KNRM()
model.params.update(preprocessor.context)
model.params['input_shapes'] = [[10,], [10, bin_size,]]  #孪生网络：zh输入维度....es输入维度
model.params['task'] = ranking_task
model.params['mask_value'] = 0  #不设置mask
model.params['embedding_output_dim'] = emb.output_dim  #emb层的输出维度 [样本个数,每个样本被pad成10个词,每个词的维度为300]
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()
print(model.params)
# #
# # # # ##字典向量构建
print('create martix ...')
embedding_matrix = emb.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
model.load_embedding_matrix(embedding_matrix)
# # #
# # ##直方图构建
print('create histogram ...')
hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=bin_size, hist_mode='LCH')
# #
# # # # ##测试集的处理,数据生成器,这里放测试集的原因是这是最终测试用的
print('test set process ...')
pred_generator = mz.DataGenerator(test_processed, mode='pair', callbacks=[hist_callback])
pred_x, pred_y = pred_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model,x=pred_x,y=pred_y,once_every=1,batch_size=len(pred_y),model_save_path=r'g:\my paper\训练的模型')
# #
# # ##训练集的处理,数据生成器
print('train set process ... ')
train_generator = mz.DataGenerator(train_processed, mode='point', num_dup=3, num_neg=10, batch_size=300,callbacks=[hist_callback])  #每个样本被采3次 弥补样本的不足
print('num batches:', len(train_generator))
history = model.fit_generator(train_generator, epochs=5, callbacks=[evaluate], workers=30, use_multiprocessing=True)
# # #
# # # ##加载训练好的模型，进行开发集的预测，并用这个结果来调参
vald=pd.read_excel(r'c:\Users\admin\Desktop\vald_file.xlsx',encoding='utf-8')
vald.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
vald_pack = mz.pack(vald)
valid_processed = preprocessor.transform(vald_pack)
print("test model's quality ...")
drmm_model = mz.load_model(r'g:\my paper\训练的模型\16')
test_generator = mz.DataGenerator(data_pack=valid_processed[:10], mode='point', callbacks=[hist_callback])
test_x, test_y = test_generator[:]
prediction = drmm_model.predict(test_x)
print(prediction)
#
