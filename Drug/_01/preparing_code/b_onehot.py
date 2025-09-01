import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


path = './_data/dacon/drug/'
train = pd.read_csv(path + 'data/train_with_atom_sequence.csv', index_col=[0,1])

# print(train.shape)  #(1681, 3)
ohe = OneHotEncoder(sparse=False)
train_ohe = ohe.fit_transform(train)
print(train_ohe.shape)
# (1681, 4478)
print(train_ohe)





