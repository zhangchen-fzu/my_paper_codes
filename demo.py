import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json
print('matchzoo version', mz.__version__)
print()

#√√√√√√√√
print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task='ranking')
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task='ranking', filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task='ranking', filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

#√√√√√√√√
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

#√√√√√√√√
print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
print("embedding loaded as `glove_embedding`")


def append_params_to_readme(model):
    import tabulate
    with open('README.rst', 'a+') as f:
        subtitle = model.params['model_class'].__name__
        line = '#' * len(subtitle)
        subtitle = subtitle + '\n' + line + '\n\n'
        f.write(subtitle)

        df = model.params.to_frame()[['Name', 'Value']]
        table = tabulate.tabulate(df, tablefmt='rst', headers='keys') + '\n\n'
        f.write(table)

#√√√√√√√√
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=100, remove_stop_words=False)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print(preprocessor.context)

#√√√√√√√√
bin_size = 30
model = mz.models.DRMM()
model.params.update(preprocessor.context)
model.params['input_shapes'] = [[10,], [10, bin_size,]]
model.params['task'] = ranking_task
model.params['mask_value'] = 0
model.params['embedding_output_dim'] = glove_embedding.output_dim
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()

#√√√√√√√√
embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
# normalize the word embedding for fast histogram generating.
l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
model.load_embedding_matrix(embedding_matrix)

#√√√√√√√√
hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')


pred_generator = mz.DataGenerator(test_pack_processed, mode='point', callbacks=[hist_callback])
pred_x, pred_y = pred_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model,x=pred_x,y=pred_y,once_every=1,batch_size=len(pred_y),model_save_path='./drmm_pretrained_model/')

train_generator = mz.DataGenerator(train_pack_processed, mode='pair', num_dup=2, num_neg=10, batch_size=200,callbacks=[hist_callback])
print('num batches:', len(train_generator))
history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)

drmm_model = mz.load_model('./drmm_pretrained_model/16')
test_generator = mz.DataGenerator(data_pack=dev_pack_processed[:10], mode='point', callbacks=[hist_callback])
test_x, test_y = test_generator[:]
prediction = drmm_model.predict(test_x)
print(prediction)

