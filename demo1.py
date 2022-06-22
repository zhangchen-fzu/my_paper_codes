import matchzoo as mz

task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=10))  #任务类型为排序任务

##加载数据
train_raw = mz.datasets.toy.load_data(stage='train', task='ranking')
test_raw = mz.datasets.toy.load_data(stage='test', task='ranking')

##数据预处理
preprocessor = mz.preprocessors.BasicPreprocessor()
preprocessor.fit(train_raw)
train_processed = preprocessor.transform(train_raw)
test_processed = preprocessor.transform(test_raw)


# vocab_unit = preprocessor.context['vocab_unit']
# print('Orig Text:', train_processed.left.loc['Q1']['text_left'])
# sequence = train_processed.left.loc['Q1']['text_left']
# print('Transformed Indices:', sequence)
# print('Transformed Indices Meaning:',
#       '_'.join([vocab_unit.state['index_term'][i] for i in sequence]))

model = mz.models.DenseBaseline()
model.params.to_frame()[['Name', 'Description', 'Value']]
model.params['task'] = task
model.params['mlp_num_units'] = 3
print(model.params)
print(preprocessor.context['input_shapes'])
model.params.update(preprocessor.context)
model.params.completed()
model.build()
model.compile()
model.backend.summary()

x, y = train_processed.unpack()
test_x, test_y = test_processed.unpack()
model.fit(x, y, batch_size=32, epochs=5)
data_generator = mz.DataGenerator(train_processed, batch_size=32)
model.fit_generator(data_generator, epochs=5, use_multiprocessing=True, workers=4)
model.evaluate(test_x, test_y)
model.predict(test_x)

model.save('my-model')
loaded_model = mz.load_model('my-model')