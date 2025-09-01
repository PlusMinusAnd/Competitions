#===========================================
#================= import ==================
#===========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import random
import numpy as np
import tensorflow as tf


import pandas as pd
import numpy as np
import os

#===========================================
#============== 시드값 고정 ================
#===========================================


seed = 126
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

#===========================================
#============== 데이터 로드 ================
#===========================================

load_path = './Hanhwa/'
save_path = './Hanhwa/data/'
os.makedirs(save_path, exist_ok=True)

samsung_df = pd.read_csv(load_path + '삼성전자 250711.csv', encoding='cp949')
hanhwa_df = pd.read_csv(load_path + '한화에어로스페이스 250711.csv', encoding='cp949')

# print(samsung_df.shape)
# print(hanhwa_df.shape)
# (4260, 17)
# (4280, 17)
# print(samsung_df.columns)
# print(hanhwa_df.columns)

#===========================================
#============== 전처리 시작 ================
#===========================================

#===========================================
#============== 컬럼명 변경 ================
column_rename_dict = {
    '일자': 'Date',
    '시가': 'Open',
    '고가': 'High',
    '저가': 'Low',
    '종가': 'Close',
    '대비': 'Change',
    'Unnamed: 6': 'Change_Sign',  # 보통 '▲', '▼' 같은 기호
    '등락률': 'Change_Rate',
    '거래량': 'Volume',
    '금액(백만)': 'Transaction_Amount_Mil',
    '신용비': 'Credit_Ratio',
    '개인': 'Individual',
    '기관': 'Institution',
    '외인(수량)': 'Foreign_Quantity',
    '외국계': 'Foreign_Institutions',
    '프로그램': 'Program_Trading',
    '외인비': 'Foreign_Ratio'
}

samsung_df.rename(columns=column_rename_dict, inplace=True)
hanhwa_df.rename(columns=column_rename_dict, inplace=True)

# ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Change_Sign',
#        'Change_Rate', 'Volume', 'Transaction_Amount_Mil', 'Credit_Ratio',
#        'Individual', 'Institution', 'Foreign_Quantity', 'Foreign_Institutions',
#        'Program_Trading', 'Foreign_Ratio'],


#===========================================
#============= 필요없는 컬럼 ===============

samsung_df = samsung_df.drop(['Change_Rate','Change'], axis=1)
hanhwa_df = hanhwa_df.drop(['Change_Rate', 'Change'], axis=1)
# print(samsung_df.columns)
# print(hanhwa_df.columns)
# ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Change_Sign',
#        'Volume', 'Transaction_Amount_Mil', 'Credit_Ratio', 'Individual',
#        'Institution', 'Foreign_Quantity', 'Foreign_Institutions',
#        'Program_Trading', 'Foreign_Ratio']

# print(samsung_df['Open'])
# print(type(samsung_df['Open']))

# exit()

#===========================================
#============== 데이터 처리 ================

samsung_df['Date'] = pd.to_datetime(samsung_df['Date'])
hanhwa_df['Date'] = pd.to_datetime(hanhwa_df['Date'])

for col in samsung_df.columns:
    if samsung_df[col].astype(str).str.contains(',').any():
        samsung_df[col] = samsung_df[col].astype(str).str.replace(',', '').astype(float)
        
for col in hanhwa_df.columns:
    if hanhwa_df[col].astype(str).str.contains(',').any():
        hanhwa_df[col] = hanhwa_df[col].astype(str).str.replace(',', '').astype(float)
        
# print(samsung_df['Open'])
# print(hanhwa_df['Open'])
# ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Change_Sign',
#        'Volume', 'Transaction_Amount_Mil', 'Credit_Ratio', 'Individual',
#        'Institution', 'Foreign_Quantity', 'Foreign_Institutions',
#        'Program_Trading', 'Foreign_Ratio']

# exit()
drop_low = pd.to_datetime('2015-07-13')

samsung_df = samsung_df[samsung_df['Date'] >= drop_low]
hanhwa_df = hanhwa_df[hanhwa_df['Date'] >= drop_low]

# print(samsung_df.shape)
# print(hanhwa_df.shape)
# exit()

#===========================================
#============== 50:1분할 처리 ==============

split_date = pd.to_datetime('2018-05-04')
pre_split = samsung_df['Date'] < split_date

devide_adjust = ['Open', 'High', 'Low', 'Close', 'Change_Sign']
multiply_adjust = ['Volume', 'Individual', 'Institution', 'Foreign_Quantity', 'Foreign_Institutions', 'Program_Trading' ]

samsung_df.loc[pre_split, devide_adjust] = samsung_df.loc[pre_split, devide_adjust] / 50
samsung_df.loc[pre_split, multiply_adjust] = samsung_df.loc[pre_split, multiply_adjust] * 50

# print(samsung_df.shape)

#===========================================
#========== 날짜를 숫자로 변환 =============

cols_except_date = samsung_df.columns.difference(['Date'])

# 해당 행에 NaN이 하나라도 있으면 → 전체 열 NaN으로 만들기 (Date 제외)
samsung_df.loc[samsung_df[cols_except_date].isna().any(axis=1), cols_except_date] = np.nan
hanhwa_df.loc[hanhwa_df[cols_except_date].isna().any(axis=1), cols_except_date] = np.nan
num_cols = samsung_df.select_dtypes(include=['number']).columns

samsung_df[num_cols] = samsung_df[num_cols].interpolate(method='linear')
hanhwa_df[num_cols] = hanhwa_df[num_cols].interpolate(method='linear')

samsung_df['Weekday'] = samsung_df['Date'].dt.day_name()
hanhwa_df['Weekday'] = hanhwa_df['Date'].dt.day_name()

df_list = [samsung_df, hanhwa_df]

for i in df_list :
    i['Year'] = i['Date'].dt.year
    i['Month'] = i['Date'].dt.month
    i['Day'] = i['Date'].dt.day

samsung_df = samsung_df.drop(columns=['Date'], axis=1)  # datetime은 제거
hanhwa_df = hanhwa_df.drop(columns=['Date'], axis=1)  # datetime은 제거

# print(samsung_df.shape)     # (4260, 18)
# print(hanhwa_df.shape)      # (4280, 18)
# print(samsung_df.columns)
# print(hanhwa_df.columns)
# Index(['Open', 'High', 'Low', 'Close', 'Change', 'Change_Sign', 'Volume',
#        'Transaction_Amount_Mil', 'Credit_Ratio', 'Individual', 'Institution',
#        'Foreign_Quantity', 'Foreign_Institutions', 'Program_Trading',
#        'Foreign_Ratio', 'Year', 'Month', 'Day'],
#       dtype='object')
# exit()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
samsung_df['Weekday'] = le.fit_transform(samsung_df['Weekday'])
hanhwa_df['Weekday'] = le.transform(hanhwa_df['Weekday'])

onehot_cols = ['Weekday', 'Year', 'Month', 'Day']

for j in onehot_cols :
    encoded = pd.get_dummies(samsung_df[j], prefix=f'{j}')
    samsung_df = pd.concat([samsung_df, encoded], axis=1)
    
for k in onehot_cols :
    encoded = pd.get_dummies(hanhwa_df[k], prefix=f'{k}')
    hanhwa_df = pd.concat([hanhwa_df, encoded], axis=1)

samsung_df = samsung_df.drop(onehot_cols, axis=1)
hanhwa_df = hanhwa_df.drop(onehot_cols, axis=1)
# print(samsung_df.columns)
# print(hanhwa_df.columns)
# print(np.unique(samsung_df['Weekday_0'], return_counts=True))
# print(np.unique(hanhwa_df['Weekday_0'], return_counts=True))
# exit()
#===========================================
#============== 결측치 처리 ================

# print(samsung_df.isna().sum())
# print(hanhwa_df.isna().sum())
# print(samsung_df.shape)
# print(hanhwa_df.shape)
# print(samsung_df.columns)
# print(hanhwa_df.columns)

# samsung_df.to_csv(save_path + 'samsung.csv', index=False)
# hanhwa_df.to_csv(save_path + 'hanhwa.csv', index=False)

# samsung_df
# (2454, 77)
# Index(['Open', 'High', 'Low', 'Close', 'Change_Sign', 'Volume',
#        'Transaction_Amount_Mil', 'Credit_Ratio', 'Individual', 'Institution',
#        'Foreign_Quantity', 'Foreign_Institutions', 'Program_Trading',
#        'Foreign_Ratio', 'Weekday', 'Year', 'Month', 'Day', 'Weekday_Friday',
#        'Weekday_Monday', 'Weekday_Thursday', 'Weekday_Tuesday',
#        'Weekday_Wednesday', 'Year_2015', 'Year_2016', 'Year_2017', 'Year_2018',
#        'Year_2019', 'Year_2020', 'Year_2021', 'Year_2022', 'Year_2023',
#        'Year_2024', 'Year_2025', 'Month_1', 'Month_2', 'Month_3', 'Month_4',
#        'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#        'Month_11', 'Month_12', 'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5',
#        'Day_6', 'Day_7', 'Day_8', 'Day_9', 'Day_10', 'Day_11', 'Day_12',
#        'Day_13', 'Day_14', 'Day_15', 'Day_16', 'Day_17', 'Day_18', 'Day_19',
#        'Day_20', 'Day_21', 'Day_22', 'Day_23', 'Day_24', 'Day_25', 'Day_26',
#        'Day_27', 'Day_28', 'Day_29', 'Day_30', 'Day_31'],
#       dtype='object')

# hanhwa_df
# (2454, 77)
# Index(['Open', 'High', 'Low', 'Close', 'Change_Sign', 'Volume',
#        'Transaction_Amount_Mil', 'Credit_Ratio', 'Individual', 'Institution',
#        'Foreign_Quantity', 'Foreign_Institutions', 'Program_Trading',
#        'Foreign_Ratio', 'Weekday', 'Year', 'Month', 'Day', 'Weekday_Friday',
#        'Weekday_Monday', 'Weekday_Thursday', 'Weekday_Tuesday',
#        'Weekday_Wednesday', 'Year_2015', 'Year_2016', 'Year_2017', 'Year_2018',
#        'Year_2019', 'Year_2020', 'Year_2021', 'Year_2022', 'Year_2023',
#        'Year_2024', 'Year_2025', 'Month_1', 'Month_2', 'Month_3', 'Month_4',
#        'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#        'Month_11', 'Month_12', 'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5',
#        'Day_6', 'Day_7', 'Day_8', 'Day_9', 'Day_10', 'Day_11', 'Day_12',
#        'Day_13', 'Day_14', 'Day_15', 'Day_16', 'Day_17', 'Day_18', 'Day_19',
#        'Day_20', 'Day_21', 'Day_22', 'Day_23', 'Day_24', 'Day_25', 'Day_26',
#        'Day_27', 'Day_28', 'Day_29', 'Day_30', 'Day_31'],
#       dtype='object')

#####################################################
#####################################################
#####################################################


def split_all(dataset, timesteps, strides):
    all = []
    for i in range(0, len(dataset) - timesteps + 1, strides) :
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all[:])
    return all

# 1. 데이터
path = './Hanhwa/data/'
# samsung = pd.read_csv(path + 'samsung.csv')
samsung = samsung_df.copy()
# hanhwa = pd.read_csv(path + 'hanhwa.csv')
hanhwa = hanhwa_df.copy()

timesteps = 5
stride = 1

#한화에어로스페이스 데이터
Xh = hanhwa.iloc[10:].drop(['Open'], axis=1)
test_h = hanhwa.iloc[5:10, 1:]  # 5행 선택
H = split_all(dataset=Xh, timesteps=timesteps, strides=stride)

Yh = hanhwa['Open'].iloc[:-14]

H = np.array(H)
test_h = np.array(test_h)
Yh = np.array(Yh)

# Yh_max = np.max(Yh)

# Yh = Yh/Yh_max

# print(H.shape)  #(489, 5, 76)
# print(H)
# print(Yh.shape) #(489,)
# print(Yh)
# print(test_h)
# exit()
h_train, h_test, y2_train, y2_test = train_test_split(
    H, Yh, random_state=seed, train_size=0.8
)

# 1. reshape
h_train_2d = h_train.reshape(-1, 5*72)
h_test_2d = h_test.reshape(-1, 5*72)
test_h_2d = test_h.reshape(-1, 5*72)

# 2. 스케일링
ss_h = StandardScaler()
h_train_scaled = ss_h.fit_transform(h_train_2d)
h_test_scaled = ss_h.transform(h_test_2d)
test_h_scaled = ss_h.transform(test_h_2d)

# 3. 다시 원래 shape로 reshape
h_train = h_train_scaled.reshape(-1, 5, 72)
h_test = h_test_scaled.reshape(-1, 5, 72)
test_h = test_h_scaled.reshape(-1, 5, 72)


#삼성전자 데이터
Xs = samsung.iloc[10:].drop(['Open'], axis=1)
test_s = samsung.iloc[5:10, 1:]  # 5행 선택
S = split_all(dataset=Xs, timesteps=timesteps, strides=stride)

Ys = samsung['Open'].iloc[:-14]

S = np.array(S)
test_s = np.array(test_s)
Ys = np.array(Ys)

# Ys = Ys / Yh_max

# print(S.shape)  #(489, 5, 76)
# print(S)
# print(Ys.shape) #(489,)
# print(Ys)
# print(test_s)

s_train, s_test= train_test_split(
    S,random_state=seed, train_size=0.8
)

# print(s_train.shape)(391, 5, 72)
# exit()

# 1. reshape
s_train_2d = s_train.reshape(-1, 5*72)
s_test_2d = s_test.reshape(-1, 5*72)
test_s_2d = test_s.reshape(-1, 5*72)

# 2. 스케일링
ss_s = StandardScaler()
s_train_scaled = ss_s.fit_transform(s_train_2d)
s_test_scaled = ss_s.transform(s_test_2d)
test_s_scaled = ss_s.transform(test_s_2d)

# 3. 다시 원래 shape로 reshape
s_train = s_train_scaled.reshape(-1, 5, 72)
s_test = s_test_scaled.reshape(-1, 5, 72)
test_s = test_s_scaled.reshape(-1, 5,72)

# print(s_train)

# print(s_train.shape)
# print(h_train.shape)
# print(s_test.shape)
# print(h_test.shape)

# exit()
############# #2-1 samsung 모델 ###############

from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, SimpleRNN, Input, ReLU, Dropout, BatchNormalization, Dense

input11 = Input(shape=(5,72))
gru11 = LSTM(256)(input11)
rel11 = ReLU()(gru11)
den11 = Dense(512, activation='relu')(rel11)
bat11 = BatchNormalization()(den11)
dro11 = Dropout(0.4)(bat11)
den12 = Dense(256, activation='relu')(dro11)
bat12 = BatchNormalization()(den12)
dro12 = Dropout(0.3)(bat12)
den13 = Dense(128, activation='relu')(dro12)
output11 = Dense(64, activation='relu')(den13)

############# #2-2 hanhwa 모델 ###############

input21 = Input(shape=(5,72))
gru21 = LSTM(256)(input21)
rel21 = ReLU()(gru21)
den21 = Dense(512, activation='relu')(rel21)
bat21 = BatchNormalization()(den21)
dro21 = Dropout(0.4)(bat21)
den22 = Dense(256, activation='relu')(dro21)
bat21 = BatchNormalization()(den22)
dro21 = Dropout(0.3)(bat21)
den23 = Dense(128, activation='relu')(dro21)
output21 = Dense(64, activation='relu')(den23)

############# #2-3 ensemble 모델 ###############

from keras.layers import Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
merge_in = Concatenate(axis=1)([output11, output21])
mer1 = Dense(256, activation='relu')(merge_in)
bat1 = BatchNormalization()(mer1)
dro1 = Dropout(0.2)(bat1)
mer2 = Dense(128, activation='relu')(dro1)
bat2 = BatchNormalization()(mer2)
dro2 = Dropout(0.1)(bat2)
mer3 = Dense(64, activation='relu')(dro2)
output3 = Dense(1)(mer3)
model = Model(inputs=[input11, input21], outputs=output3)

import datetime, os
es = EarlyStopping(
    monitor='val_loss', mode='min', restore_best_weights=True, patience=40
)
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './Hanhwa/mcp/'
os.makedirs(path1, exist_ok=True)
filename = '({epoch:04d}-{val_loss:.4f}).hdf5'
filepath = ''.join([path1, 'mcp_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath,
    verbose=1
)

rl = ReduceLROnPlateau(
    monitor='val_loss',       
    factor=0.2, 
    patience=5, 
    verbose=1,  
    mode='min', 
    min_delta=1e-4,
    cooldown=5,    
    min_lr=1e-6  
)


model.compile(loss='mae', optimizer='adam', metrics='mae')
model.fit([s_train, h_train], y2_train, epochs=10000, 
          batch_size=32, validation_split=0.2,
          callbacks=[es, mcp])

loss = model.evaluate([s_test, h_test], y2_test)
result = model.predict([s_test, h_test])
mae = mean_absolute_error(y2_test, result)

pred_h = model.predict([test_s, test_h])
# pred_h = pred_h * Yh_max
pred_h_rint = np.round(pred_h)
pred_scalar = pred_h_rint.item()
print(f'SEED : {seed}')
print('hanhwa 예측값:', pred_scalar)
print("MAE:", mae)

import shutil
low_SCORE_THRESHOLD = 800000
high_SCORE_THRESHOLD = 815000

if pred_scalar < low_SCORE_THRESHOLD:
    shutil.rmtree("./Hanhwa/mcp/")
    print(f"🚫 예상 시가 {pred_scalar} < 기준 {low_SCORE_THRESHOLD} \n→ 전체 디렉토리 삭제 완료")
elif pred_scalar > high_SCORE_THRESHOLD :
    shutil.rmtree("./Hanhwa/mcp/")
    print(f"🚫 예상 시가 {pred_scalar} > 기준 {high_SCORE_THRESHOLD} \n→ 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 예상 시가 {low_SCORE_THRESHOLD}<{pred_scalar}<{high_SCORE_THRESHOLD} \n→ 디렉토리 유지")


# SEED : 126
# hanhwa 예측값: 811288.0
# MAE: 7195.311535682633