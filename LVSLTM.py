import torch
import numpy as np
import pandas as pd
import matchzoo as mz
#
# #定义任务类型和评价指标
print('task define ...')
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=3))  #任务类型为排序任务
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

##模型参数赋值
model = mz.models.MVLSTM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 300
model.params['lstm_units'] = 20
model.params['top_k'] = 5
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 5
model.params['mlp_activation_func'] = 'relu'
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adadelta'
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()

##字典向量构建
embedding_matrix = emb.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

##测试集的处理,数据生成器,这里放测试集的原因是这是最终测试用的
pred_x, pred_y = test_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))

##训练集的处理,数据生成器
train_generator = mz.DataGenerator(train_processed,mode='pair',num_dup=2,num_neg=3,batch_size=20)
print('num batches:', len(train_generator))
history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)