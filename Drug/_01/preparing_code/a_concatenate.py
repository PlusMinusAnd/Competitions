from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

path = './_data/dacon/drug/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

train_features = pd.read_csv(path + 'data/derived_features.csv', index_col=0)
test_features = pd.read_csv(path + 'data/derived_features_test.csv', index_col=0)

derived_train = pd.read_csv(path + 'data/derived_features.csv', index_col=0)
derived_test = pd.read_csv(path + 'data/derived_features_test.csv', index_col=0)

ring_train = pd.read_csv(path + 'data/no_ring_formula_train.csv', index_col=0)
ring_test = pd.read_csv(path + 'data/no_ring_formula_test.csv', index_col=0)

train = pd.concat([train_csv, derived_train, ring_train], axis=0)
test = pd.concat([test_csv, derived_test, ring_test], axis=0)

print(train.shape)  #(1681, 12)
print(test.shape)   #(100, 11)

train.to_csv(path+'data/06_25_train.csv')
test.to_csv(path+'data/06_25_test.csv')
















