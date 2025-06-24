from datasets import load_dataset
import pandas as pd

categories = ['astronomy', 'econometrics', 'world_religions']
seed = 42
set_size = 50

for category in categories:

    ds = load_dataset("cais/mmlu", category, split='test')
    shuffled_ds = ds.shuffle(seed=seed)
    shuffled_ds = shuffled_ds[:set_size]
    shuffled_ds = pd.DataFrame(shuffled_ds)
    shuffled_ds.to_csv(f'mmlu_data/{category}_questions.csv', index=False)