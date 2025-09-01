import pandas as pd
import numpy as np

train = np.load('./Drug/_npy_data/train_graph.npy')
test = np.load('./Drug/_npy_data/test_graph.npy')

# print(train.shape)  (1681, 60, 60, 2)
# print(test.shape)   (100, 60, 60, 2)   

flat_train = train.flatten()
flat_test = test.flatten()

unique_trn, counts_trn = np.unique(flat_train, return_counts=True)
unique_tst, counts_tst = np.unique(flat_test, return_counts=True)

value_counts_1 = dict(zip(unique_trn, counts_trn))
value_counts_2 = dict(zip(unique_tst, counts_tst))
# print(value_counts_1)   {0.0: 11925916, 1.0: 128486, 1.5: 42858, 2.0: 5720, 3.0: 220}
# print('###############')
# print(value_counts_2)   {0.0: 709416, 1.0: 7594, 1.5: 2716, 2.0: 266, 3.0: 8}

# [..., 0]: 결합 여부 (0 또는 1)
# [..., 1]: 결합 강도 (단일=1, 이중=2, 삼중=3, 방향족=1.5)
