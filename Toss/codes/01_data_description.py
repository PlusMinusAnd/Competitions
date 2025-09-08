import pandas as pd
import numpy as np

train = pd.read_parquet("./Project/Toss/_split/train_200k/part-00000.parquet")
print(train.shape)
test = pd.read_parquet("./Project/Toss/_split/test_200k/part-00000.parquet")
print(test.shape)

print(list(set(test.columns)-set(train.columns)))

test = test.drop(['ID'], axis=1)
print(test.shape)

print(train.describe)
print(test.describe)
print(train.isna().sum())
print(test.isna().sum())
print(train.head(10))
print(test.head(10))
print(pd.value_counts(train['gender']))