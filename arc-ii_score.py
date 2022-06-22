import torch
import numpy as np
import pandas as pd
import matchzoo as mz
#
# #定义任务类型和评价指标
print('task define ...')
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=10))  #任务类型为排序任务
ranking_task.metrics = [mz.metrics.MeanAveragePrecision()]  #最终的评价指标为MAP

# #加载自己的嵌入向量
print("embedding load ...")
emb = mz.embedding.load_from_file(r'g:\my paper\预对齐嵌入向量\all_align_quchong.vec',mode='fasttext')

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

# #数据预处理
print('data preproces ...')
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=10, remove_stop_words=False)
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(vald_pack)
test_processed = preprocessor.transform(test_pack)

##模型设置
model = mz.models.KNRM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 100
model.params['embedding_trainable'] = True
model.params['num_blocks'] = 2
model.params['kernel_1d_count'] = 32
model.params['kernel_1d_size'] = 3
model.params['kernel_2d_count'] = [64, 64]
model.params['kernel_2d_size'] = [3, 3]
model.params['pool_2d_size'] = [[3, 3], [3, 3]]
model.params['optimizer'] = 'adam'
model.build()
model.compile()
print(model.params)

# # # # ##字典向量构建
print('create martix ...')
embedding_matrix = emb.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

##测试集的处理
test_x, test_y = test_processed[:].unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=test_x, y=test_y, batch_size=len(test_y))

##训练集的生成及训练
train_generator = mz.DataGenerator(train_processed,mode='pair',num_dup=2,num_neg=1,batch_size=20)
print('num batches:', len(train_generator))
history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)
