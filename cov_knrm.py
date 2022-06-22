import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json
print('matchzoo version', mz.__version__)
print()

# #定义任务类型和评价指标
print('task define ...')
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=5))  #任务类型为排序任务
ranking_task.metrics = [mz.metrics.MeanAveragePrecision()]  #最终的评价指标为MAP

#加载自己的嵌入向量
print("embedding load ...")
emb = mz.embedding.load_from_file(r'g:\my paper\预对齐嵌入向量\all_align_quchong.vec',mode='fasttext')

# #加载数据，转换数据类型，划分训练集和测试集，定义数据
print('data loading ...')
train = pd.read_excel(r'c:\Users\admin\Desktop\train_file.xlsx',encoding='utf-8')
train.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
test=pd.read_excel(r'c:\Users\admin\Desktop\test_file.xlsx',encoding='utf-8')
test.rename(columns={'zh_term': 'text_left', 'es_term': 'text_right','biaoqian':'label'}, inplace=True)
train_pack = mz.pack(train)
test_pack=mz.pack(test)

# #数据预处理
print('data preproces ...')
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=10, remove_stop_words=False)
train_processed = preprocessor.fit_transform(train_pack)
test_processed = preprocessor.transform(test_pack)

# # # # # ##模型参数赋值
model = mz.models.ConvKNRM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = emb.output_dim
model.params['embedding_trainable'] = True
model.params['filters'] = 128
model.params['conv_activation_func'] = 'tanh'
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True
model.params['kernel_num'] = 8
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()
print(model.params)

##字典向量构建
print('create martix ...')
embedding_matrix = emb.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

##测试集的处理,数据生成器,这里放测试集的原因是这是最终测试用的
pred_x, pred_y = test_processed.unpack()
print(pred_x,len(pred_x))
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x),model_save_path=r'/content/gdrive/MyDrive/app/')

##训练集的处理,数据生成器
train_generator = mz.DataGenerator( train_processed, mode='pair', num_dup=2, num_neg=5, batch_size=32)
print('num batches:', len(train_generator))
history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)