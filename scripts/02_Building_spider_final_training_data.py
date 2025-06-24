import pandas as pd

training_data_paths = ['spider_data/spider_data/train_spider.json', 'spider_data/spider_data/train_others.json']

names = ['db_id', 'query', 'question']
df_final_training_data = pd.concat([pd.read_json(training_data_paths[0]), pd.read_json(training_data_paths[1])])
df_final_training_data = df_final_training_data[names]
df_final_training_data.reset_index(drop=True, inplace=True)
df_final_training_data.to_json('spider_data/spider_data/train_final.json', index=False, orient='records')