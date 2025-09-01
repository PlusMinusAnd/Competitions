version = 'prototype'
# === TabPFN(v2)로 학습/예측 (10k 제한 자동 대응) ===
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 123
SAVE_DIR = f'./Service/{SEED}_submission'

np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

def load_data(train_path, test_path, submission_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    submission = pd.read_csv(submission_path)
    return train, test, submission

def basic_features(df) :
    df = df.copy()
    df['gender'] = df['gender'].map({'F':0, 'M':1})
    df['subscription_type'] = df['subscription_type'].map({'member':0, 'plus':1, 'vip':2})
    df['contract_length'] = df['contract_length'].map({'30':0, '90':1, '360':2})
    
    return df

train, test, submission = load_data(
    './Service/train.csv', './Service/test.csv', './Service/sample_submission.csv'
)

train = basic_features(train)
test = basic_features(test)

# print(train.columns, train.shape)
# print(test.columns, test.shape)
# ['age', 'gender', 'tenure', 'frequent', 'payment_interval',
#        'subscription_type', 'contract_length', 'after_interaction',
#        'support_needs'],
# (30858, 9)

# print(pd.value_counts(train['subscription_type']))
# 1    10481
# 2    10405
# 0     9972
