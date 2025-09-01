import pandas as pd

train = pd.read_csv('./hackathon/data/population.csv', encoding='cp949')

print(train)
print(train.shape)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. 자치구와 행정동을 합친 새로운 열 생성
train['지역코드'] = train['자치구'] + '_' + train['행정동']

# 2. 텍스트 토크나이징
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['지역코드'])
sequences = tokenizer.texts_to_sequences(train['지역코드'])

# 3. 패딩 (길이가 1이지만 일반화 위해 적용)
padded = pad_sequences(sequences)

# 4. 각 지역의 임베딩 결과 형태 확인용 (단어 수, 토큰 수, 시퀀스)
vocab_size = len(tokenizer.word_index) + 1  # Padding 포함

padded[:5], vocab_size, tokenizer.word_index

# TensorFlow 없이 단순하게 숫자 ID만 추출
from sklearn.preprocessing import LabelEncoder

# 자치구_행정동 합치기
train['자치구_행정동'] = train['자치구'] + '_' + train['행정동']

# 수치 인코딩
encoder = LabelEncoder()
train['지역코드'] = encoder.fit_transform(train['자치구_행정동'])

# '행정동' 다음, '소계_인구' 앞에 '지역코드' 삽입
cols = train.columns.tolist()
insert_at = cols.index('소계_인구')
reordered_cols = cols[:insert_at] + ['지역코드'] + cols[insert_at:]
pop_df = train[reordered_cols]

# 저장
output_file = "./hackathon/data/population_with_encoded_location.csv"
pop_df.to_csv(output_file, index=False, encoding='utf-8-sig')

output_file
