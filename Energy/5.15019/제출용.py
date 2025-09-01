version = "LETS_GO"

print(f"[{version}] ACTIVATE\n")

#region IMPORT
# ========== Standard Library ==========
import json
print(json.__version__) #2.0.9
import os
import random
import warnings
import hashlib
import datetime

# ========== Third-Party ==========
import numpy as np
print(np.__version__)   #1.26.4
import pandas as pd
print(pd.__version__)   #2.1.4

# Progress
import tqdm
print(tqdm.__version__) #4.67.1
from tqdm import tqdm

# Scikit-learn
import sklearn
print(sklearn.__version__)  #1.4.2
from sklearn.model_selection import TimeSeriesSplit

# Gradient Boosting
import xgboost
print(xgboost.__version__)  #3.0.2
from xgboost import XGBRegressor
import lightgbm
print(lightgbm.__version__) #4.6.0
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Optuna
import optuna
print(optuna.__version__)   #4.4.0

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 775
print(f"[Current Run SEED]: {SEED}")

log_path = f'./Energy/03/log/'
save_path = f'./Energy/03/{SEED}_submission_{version}/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

py_path = './Energy/03/'

N_SPLITS = 5
N_TRIALS = 25
N_SPLITS_MODEL = 5
N_TRIALS_MODEL = 25
SAMPLE_SIZE = 0.5
SAMPLE_SIZE_MODEL = 0.5
SAMPLE_SIZE_BNUM = 0.7
AUGMENTATION = 4

print(f"[Preprocessing Variable]")
print(f"N_SPLITS {N_SPLITS} | N_TRIALS {N_TRIALS}")
print(f"SAMPLE_SIZE {SAMPLE_SIZE} | AUGMENTATION {60/AUGMENTATION} times")
print(f"[Model Variable]")
print(f"N_SPLITS_MODEL {N_SPLITS_MODEL} | N_TRIALS_MODEL {N_TRIALS_MODEL}")
print(f"SAMPLE_SIZE_MODEL {SAMPLE_SIZE_MODEL} | SAMPLE_SIZE_BNUM {SAMPLE_SIZE_BNUM}\n")

#endregion 

""" 
버전 태깅- version : 콘솔 메시지와 저장 폴더 이름에 사용해, 실험별 산출물 구분.
필수 라이브러리 로드: 표준/서드파티 의존성 일괄 import.
로그/저장 경로 준비: 실행 시 폴더를 자동 생성하여 파일 충돌/덮어쓰기 방지.
재현성 보장 시드 고정: SEED를 콘솔에 표시하고, PYTHONHASHSEED, random, numpy에 동일 시드 주입.
(참고) XGBoost/LightGBM 모델 내부 시드는 모델 생성 시 random_state/seed로 별도 지정 필요.
Optuna/경고 설정: 탐색 로그를 조용히(WARNING) 만들고, 불필요한 경고를 억제.
전역 하이퍼파라미터 선언: 
1) N_SPLITS, N_TRIALS: 전처리/보조 모델 단계의 CV 분할 수 및 탐색 횟수
2) N_SPLITS_MODEL, N_TRIALS_MODEL: 본 모델 단계의 CV/탐색 설정(별도 운용)
3) SAMPLE_SIZE, SAMPLE_SIZE_MODEL, SAMPLE_SIZE_BNUM: Optuna 목적함수에서 사용할 샘플링 비율(속도·일반화 균형)
4) AUGMENTATION: 전처리 단계의 증강 강도(예: 시간 리샘플/파생 시 간격 결정 등). 60/AUGMENTATION로 사람이 보기 쉽게 요약 출력.
 """

print("[STEP 1] PREPROCESSING") # 파생 피쳐 제작 및 전처리

#region STEP 1

#region FUNCTIONS

""" 
평가지표 함수
 """
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

""" 
train 데이터와 test 데이터와 building_info, sample_submission을 불러오는 함수
1) 한국어로 된 컬럼명은 사용하기 불편하여 영어로 매핑
2) building_info와 train, test를 건물번호 기준으로 merge까지 한번에 수행하고 train, test, submission을 반환
 """
def load_data(train_path, test_path, building_path, sub_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    submission = pd.read_csv(sub_path)
    building_info = pd.read_csv(building_path)

    mapping = {
        '건물번호':'building_num',
        '기온(°C)':'temperature',
        '습도(%)':'humidity',
        '풍속(m/s)':'windspeed',
        '강수량(mm)':'precipitation',
        '일조(hr)':'sunshine',
        '일사(MJ/m2)':'solar',
        '전력소비량(kWh)':'power_consumption',
    }

    mapping_building = {
        '건물번호':'building_num',
        '건물유형': 'building_type',
        '연면적(m2)': 'all_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'pvc',
        'ESS저장용량(kWh)': 'ess',
        'PCS용량(kW)': 'pcs',
    }
    
    def apply_mapping(df):
        return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    train = apply_mapping(train)
    test = apply_mapping(test)
    
    building_info = building_info.rename(columns=mapping_building)
    
    building_col = ['pvc', 'ess', 'pcs']
    for i in building_col:
        building_info[i] = building_info[i].replace('-', np.nan).astype(float)
    building_info = building_info.fillna(0)
    
    train = pd.merge(train, building_info, on='building_num', how='left')
    test = pd.merge(test, building_info, on='building_num', how='left')

    return train, test, submission

""" 
날짜 관련 파생피쳐 제작
1) 풍속과 습도는 0일 수 없기 때문에, 0값을 모두 전후값을 바탕으로 선형 보간
 """
def date_features(df):
    df = df.copy()

    df['date'] = pd.to_datetime(df['일시'])

    df['minute'] = df['date'].dt.minute
    df['hour'] = df['date'].dt.hour       
    df['dow'] = df['date'].dt.dayofweek   
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    df = df.sort_values(['building_num', 'date']).reset_index(drop=True)

    df[['windspeed','humidity']] = df[['windspeed','humidity']].replace(0, np.nan)

    def fill_mean_ffill(s: pd.Series):
        s = s.interpolate(method='linear', limit_direction='both')
        s = s.ffill().bfill()
        return s

    df[['windspeed','humidity']] = (
        df.groupby('building_num')[['windspeed','humidity']]
        .transform(fill_mean_ffill)
    )

    
    return df

""" 
다양한 파생피쳐 제작
1) 주기성 정보에는 SIN, COS 피쳐를 제작
2) PT, DI, CDH 등 여러 관련 파생피쳐를 제작
3) 데이터를 확인해본 결과 같은 일시에 날씨가 정확하게 동일한 건물들이 있는 것을 확인, 
   그 건물들을 building_group으로 표시
 """
def feature_engineering(df):
    df = df.copy()

    _total_min = ((pd.to_numeric(df['hour'], errors='coerce') * 60) +
                pd.to_numeric(df['minute'], errors='coerce')) % 1440

    _theta = 2 * np.pi * (_total_min / 1440.0)

    df['SIN_Time'] = np.sin(_theta).astype('float32')
    df['COS_Time'] = np.cos(_theta).astype('float32')
    df['SIN_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['COS_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    days_in_month = df['month'].map({6: 30, 7: 31, 8: 31}).astype('int16')

    theta = 2 * np.pi * (df['day'] - 1) / days_in_month

    df['SIN_day'] = np.sin(theta).astype('float32')
    df['COS_day'] = np.cos(theta).astype('float32')
        
    df['SIN_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['COS_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['SIN_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['COS_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    
    df['SIN_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['COS_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    df = df.drop(['temp_date'], axis=1)
    
    temp = df['temperature']
    humidity = df['humidity']
    wind_speed = df['windspeed']
    df['PT'] = temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00
    df['CDH'] = np.maximum(df['temperature'] - 26, 0)
    df['DI'] = 0.81 * df['temperature'] + 0.01 * df['humidity'] * (0.99 * df['temperature'] - 14.3) + 46.3
    df['PVC_per_CA'] = df['pvc'] / (df['cooling_area'] + 1e-6)
    df['ESS_installation'] = df['ess'].apply(lambda x: 1 if x > 0 else 0)
    df['PCS_installation'] = df['pcs'].apply(lambda x: 1 if x > 0 else 0)
    df['Facility_Density'] = (df['ess'] + df['pcs']) / (df['all_area'] + 1e-6)
    
    yr = df['date'].dt.year.astype(str)
    season_start = pd.to_datetime(yr + "-06-01")
    season_end   = pd.to_datetime(yr + "-09-01")
    in_season = (df['date'] >= season_start) & (df['date'] < season_end)
    period_sec = (season_end - season_start).dt.total_seconds()
    pos = (df['date'] - season_start).dt.total_seconds() / period_sec  # 0~1
    phi = 2 * np.pi * pos
    df['SIN_summer'] = np.where(in_season, np.sin(phi), np.nan)
    df['COS_summer'] = np.where(in_season, np.cos(phi), np.nan)

    building_groups = [
            [28], [72], [19,58,75,91], [77], [24], [61,74,81], [32,42,65,79,99],
            [11,12,13,41,68,83,88], [20,26,44,45,70,100], [1,2,3,4,5,6,7,8,27,33,34,35,37,47,67,86,96],
            [71], [54,84], [17,18,29,30,31,40,43,48,49,51,52,53,60,63,64,76,78], [66], [85],
            [55,82], [15,16,39,59,73,92], [80,87], [89,90], [98], [50], [21,22,23],
            [46,93,94,95], [14,69], [57], [97], [36,38,56], [25,62], [9,10]
        ]
    
    building_to_group = {}
    for group_id, buildings in enumerate(building_groups):
        for building in buildings:
            building_to_group[building] = group_id

    def add_region_group(x):
        x = x.copy()
        x['groupID'] = x['building_num'].map(building_to_group)
        return x

    df = add_region_group(df)
    
    return df

""" 
데이터 증강함수
1) numeric_cols를 자동으로 탐색하고 선택하여 minutes 간격으로 보간
2) numeric_cols에 선택되지 않는 컬럼은 전 값을 그대로 복사
 """
def upsample_20min_linear(
    df: pd.DataFrame,
    time_col: str,
    group_cols=None,
    numeric_cols=None,
    copy_non_numeric_from='prev',
    enforce_hour_gap=False,
    minutes=15
) -> pd.DataFrame:
    df = df.copy()
    group_cols = group_cols or []

    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        try:
            df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
        except Exception:
            df[time_col] = pd.to_datetime(df[time_col])

    # 수치 컬럼 자동 선택
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.difference(group_cols).tolist()

    non_num_cols = [c for c in df.columns if c not in numeric_cols + [time_col]]

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).reset_index(drop=True)
        rows = []
        n = len(g)
        for i in range(n - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]

            # 원본 행
            base = cur.copy()
            base['is_interpolated'] = 0
            rows.append(base)

            delta = nxt[time_col] - cur[time_col]
            if enforce_hour_gap and delta != pd.Timedelta(hours=1):
                continue

            step = pd.Timedelta(minutes=minutes)
            m = int(delta / step)

            for j in range(1, m):
                t = cur[time_col] + j * step
                frac = (t - cur[time_col]) / delta

                new = cur.copy()
                new[time_col] = t
                for col in numeric_cols:
                    new[col] = cur[col] + float(frac) * (nxt[col] - cur[col])
                new['is_interpolated'] = 1

                if copy_non_numeric_from == 'next':
                    for col in non_num_cols:
                        new[col] = nxt[col]
                elif copy_non_numeric_from is None:
                    for col in non_num_cols:
                        new[col] = pd.NA

                rows.append(new)

        last = g.iloc[-1].copy()
        last['is_interpolated'] = 0
        rows.append(last)
        return pd.DataFrame(rows)

    if group_cols:
        out = (df.groupby(group_cols, group_keys=False)
                 .apply(_one_group)
                 .sort_values(group_cols + [time_col])
                 .reset_index(drop=True))
    else:
        out = _one_group(df).sort_values(time_col).reset_index(drop=True)

    out['minute'] = out[time_col].dt.minute.astype('int16')
    return out

""" 
전력소비량이 가장 높은 날과 전력사용량이 가장 낮은 날을 직접 라벨링
1) peak가 있었는데 성능에 악영향을 주는 것을 발견하고 제거되어 지금은 holidays만 남음
2) 직접 train의 전력 소비량을 시각화하여 확인하고 holidays로 의심되는 요일을 직접 선택
3) 주별로 규칙적인 휴일이 없는 경우에는 따로 holidays를 날짜로 표시
4) train의 규칙성을 보고 test의 휴일을 유추함(ex. 매달 둘째 넷째 월요일 휴무하는 건물은 8월 26일에 휴일로 예상) 
 """
def peak_holidays(df, is_train=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['일시'])
    df['dow']  = df['date'].dt.weekday

    df['holidays'] = 0

    df.loc[(df['building_num']==2) & (df['dow']==5), 'holidays'] = 1
    df.loc[(df['building_num']==3) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==5) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==6) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==7) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['building_num']==8) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==12) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==13) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==14) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==15) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==16) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==17) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==18) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['building_num']==20) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==21) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==22) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==23) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==24) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # 33?
    df.loc[(df['building_num']==37) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==38) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==39) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==42) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==43) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==44) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==46) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==47) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==48) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==49) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==51) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==52) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==53) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==55) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==60) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==61) & (df['dow'].isin([1,2,3,4,5])), 'holidays'] = 1
    df.loc[(df['building_num']==62) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==66) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==67) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==68) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==69) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==72) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==75) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==80) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==82) & (df['dow'].isin([0])), 'holidays'] = 1
    df.loc[(df['building_num']==83) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==86) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==87) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==90) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==94) & (df['dow'].isin([5,6])), 'holidays'] = 1

    if is_train :
        nat = df['date'].dt.strftime('%m-%d').isin(['06-06','08-15'])
        df.loc[nat, 'holidays'] = 1

        # 33은 이상하니까
        df.loc[(df['building_num']==33) & (df['date'].dt.strftime('%m-%d').isin(['06-08','07-05','07-06','07-07','08-23'])), 'holidays'] = 2
        
        df.loc[(df['building_num']==19) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-23','07-10','07-28','08-10'])), 'holidays'] = 2
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-08','07-22','08-12'])), 'holidays'] = 2
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==45) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==54) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01'])), 'holidays'] = 2
        df.loc[(df['building_num']==79) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==95) & (df['date'].dt.strftime('%m-%d').isin(['07-08','08-05'])), 'holidays'] = 2
    
    else :
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 2
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 2        
    
    return df

""" 
직접 시각화한 자료를 확인하여 이상치를 기간과 단일 시간으로 측정하여 제거함
1) 여러 이상치 처리 방법에서 이상치로 확인 되지 않지만 시각화에서 이상치로 보이는 것들이 많았음
2) IQR과 같은 여러 이상치 처리 방식을 사용해 봤으나, 결과적으로 직접 제거하는 것이 효능이 확실함을 확인하였음
 """
intervals = {
    5: [(2024080407, 2024080408)], #
    6: [(2024081500, 2024081900)],
    7: [(2024070710, 2024070811), (2024071214, 2024080603)],
    8: [(2024072108, 2024072111)], #
    9: [(2024061210, 2024061211)], #
    12: [(2024072109, 2024072111), (2024082408, 2024082410)], #
    17: [(2024062515, 2024062609)],
    19: [(2024073113, 2024073116)], #
    20: [(2024060110, 2024060111)], #
    25: [(2024070412, 2024070414)], #
    26: [(2024061714, 2024061811)], #
    28: [(2024071714, 2024071715), (2024060906, 2024060920)], #
    29: [(2024061522, 2024061523), (2024062700, 2024062701)],
    36: [(2024060100, 2024060923)],
    38: [(2024071714, 2024071715)], #
    40: [(2024071400, 2024071401)],
    41: [(2024062201, 2024062204), (2024071714, 2024071715)],
    43: [(2024061017, 2024061018), (2024081216, 2024081217)],
    44: [(2024060612, 2024060613), (2024063000, 2024063002)], #
    52: [(2024081000, 2024081002)],
    53: [(2024061417, 2024061707), (2024081816, 2024081907)],
    57: [(2024060100, 2024060721)],
    60: [(2024071714, 2024071715)], #
    62: [(2024071714, 2024071715)], #
    65: [(2024060100, 2024060823)], #
    67: [(2024061017, 2024061018), (2024072600, 2024072800), (2024080115, 2024080116), (2024081216, 2024081217)], #
    68: [(2024062823, 2024062901)],
    69: [(2024071714, 2024071715)], #
    70: [(2024060409, 2024060509)], #
    72: [(2024061100, 2024061102), (2024072110, 2024072111)],
    76: [(2024062013, 2024062016)], #
    78: [(2024071712, 2024071713)], #
    79: [(2024081903, 2024081905)],
    80: [(2024070609, 2024070615), (2024070811, 2024070820), (2024072009, 2024072013)], #
    88: [(2024082306, 2024082308)],
    89: [(2024071208, 2024071210)], #
    90: [(2024060517, 2024060518)], #
    92: [(2024071714, 2024071723)], #
    94: [(2024072620, 2024080507)],
    95: [(2024080510, 2024080511)], #
    97: [(2024071713, 2024071715)], #
    98: [(2024061314, 2024061315)], #
    99: [(2024071005, 2024071007)],
}

singles = {
    3: [2024071714], #
    12: [2024071714], #
    18: [2024061117, 2024071714, 2024080815], #
    30: [2024071320, 2024072500],
    31: [2024071714], #
    42: [2024071714], #
    46: [2024071714], #
    47: [2024071714], #
    50: [2024070514, 2024080815], #
    51: [2024072917], #
    55: [2024071714], #
    57: [2024062104], #
    73: [2024070822],
    76: [2024060313, 2024082221], #
    77: [2024080617],
    78: [2024071714],
    81: [2024062714, 2024071714], #
    82: [2024071714], #
    83: [2024071714], #
}

""" 
<핵심 아이디어>
train 데이터로만 그룹 통계를 만들고 → 2) test에 계층적 백오프로 머지
복잡도 높은 시계열 모델 없이도 “건물/시각/요일/휴일”의 규칙적 패턴을 강하게 반영
<동작 방식>
글로벌 통계 계산: global_mean, global_std
요일 비율(ratio)을 적용해 __pc(보정 타깃) 생성

그룹 통계 집계:
g1: (건물, 시각, 요일) → dow_hour_mean, dow_hour_std
g2: (건물, 시각, 휴일) → holiday_mean, holiday_std
g3: (건물, 시각) → hour_mean, hour_std
gb: (건물) → building_mean, building_std

dow_hour_mean → 없으면 holiday_mean → hour_mean → building_mean → global_mean
표준편차 계열도 동일한 우선순위로 채움
 """
def build_stats_features(
    train, test,
    target_col='power_consumption',
    building_col='building_num',
    hour_col='hour', dow_col='dow', holiday_col='holidays',
    ddof=1,
    apply_dow_ratio=True, 
    mode='byb'
):
    tr = train.copy()
    te = test.copy()

    global_mean = tr[target_col].mean()
    global_std  = tr[target_col].std(ddof=ddof)

    tr_feat = tr.copy()
    
    pc = tr_feat[target_col].to_numpy(dtype=float)
    if apply_dow_ratio:
        ratio = np.array([0.985, 0.98, 0.98, 0.995, 0.995, 0.99, 0.99], dtype=float)
        if mode == 'all':
            ratio = ratio - 0.005
        idx = tr_feat[dow_col].to_numpy(dtype=int)
        pc = pc * ratio[idx]

    tmp = tr_feat[[building_col, hour_col, dow_col, holiday_col]].copy()
    tmp['__pc'] = pc

    g1 = (tmp.groupby([building_col, hour_col, dow_col], as_index=False)
            .agg(dow_hour_mean=('__pc', 'mean'),
                dow_hour_std =('__pc', lambda x: x.std(ddof=ddof))))

    g2 = (tmp.groupby([building_col, hour_col, holiday_col], as_index=False)
            .agg(holiday_mean=('__pc', 'mean'),
                holiday_std =('__pc', lambda x: x.std(ddof=ddof))))

    g3 = (tmp.groupby([building_col, hour_col], as_index=False)
            .agg(hour_mean=('__pc', 'mean'),
                hour_std =('__pc', lambda x: x.std(ddof=ddof))))

    gb = (tmp.groupby([building_col], as_index=False)
            .agg(building_mean=('__pc', 'mean'),
                building_std =('__pc', lambda x: x.std(ddof=ddof))))

    def _attach(df):
        out = df.merge(g1, on=[building_col, hour_col, dow_col], how='left')
        out = out.merge(g2, on=[building_col, hour_col, holiday_col], how='left')
        out = out.merge(g3, on=[building_col, hour_col], how='left')
        out = out.merge(gb, on=[building_col], how='left')
        return out

    tr = _attach(tr)
    te = _attach(te)

    for df in (tr, te):
        df['dow_hour_mean'] = (
            df['dow_hour_mean']
              .fillna(df['holiday_mean'])
              .fillna(df['hour_mean'])
              .fillna(df['building_mean'])
              .fillna(global_mean)
        )
        df['dow_hour_std'] = (
            df['dow_hour_std']
              .fillna(df['holiday_std'])
              .fillna(df['hour_std'])
              .fillna(df['building_std'])
              .fillna(global_std)
        )
        df['holiday_mean'] = df['holiday_mean'].fillna(df['hour_mean']).fillna(df['building_mean']).fillna(global_mean)
        df['holiday_std']  = df['holiday_std'].fillna(df['hour_std']).fillna(df['building_std']).fillna(global_std)
        df['hour_mean']    = df['hour_mean'].fillna(df['building_mean']).fillna(global_mean)
        df['hour_std']     = df['hour_std'].fillna(df['building_std']).fillna(global_std)

    return tr, te

""" 
<목적>
날짜/위치 기반의 태양 기하(일출·일몰·태양 고도) 를 근사 계산해, 주간/야간 구분과 태양고도 같은 물리적 신호를 피처로 제공
PV/일사량/전력수요와의 상관을 높여 모델의 물리적 일관성을 유도

<입력/가정>
필수 컬럼:
일시(문자열 yyyymmddHH… 형태; day_of_year가 없을 경우 여기서 추출)
hour(0~23 실수/정수; 시 단위)
파라미터: 위도/경도(기본: 서울), tz_hours(시간대; 한국=9)
DST(서머타임)는 고려하지 않음(필요 시 tz_hours 조정)

생성되는 주요 컬럼
sunrise_hour, sunset_hour : 시 단위의 일출·일몰 시각(현지 “시계 시간”)
daylight_prev, daylight_instant, daylight_next : 시간 경계 전후의 주간 여부 플래그
prev: [t-1, t) 구간이 주간과 겹치면 1
instant: 시각 t가 주간이면 1 (sunrise ≤ t < sunset)
next: [t, t+1) 구간이 주간과 겹치면 1
solar_elevation : 시각 t-0.5(한 시간의 중간 시점) 기준 태양 고도각(°)
 """
def attach_solar_geometry(df, *, 
                          lat=37.5665,
                          lon=126.9780, 
                          tz_hours=9.0): 
    df = df.copy()

    if 'day_of_year' not in df.columns:
        temp_date = pd.to_datetime(df['일시'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df['day_of_year'] = temp_date.dt.dayofyear
    n = df['day_of_year'].to_numpy(dtype=float)

    delta_deg = 23.44 * np.sin(np.deg2rad(360.0 * (284.0 + n) / 365.0))
    delta = np.deg2rad(delta_deg)

    B = 2.0 * np.pi * (n - 81.0) / 364.0
    eot_min = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    LSTM = 15.0 * tz_hours
    tc_min = 4.0 * (lon - LSTM) + eot_min
    tc_h = tc_min / 60.0 

    phi = np.deg2rad(lat)
    cosH0 = -np.tan(phi) * np.tan(delta)
    H0 = np.arccos(np.clip(cosH0, -1.0, 1.0)) 
    H0h = (12.0 / np.pi) * H0  
    sunrise_solar = 12.0 - H0h
    sunset_solar  = 12.0 + H0h

    sunrise = sunrise_solar - tc_h
    sunset  = sunset_solar  - tc_h

    df['sunrise_hour'] = sunrise
    df['sunset_hour']  = sunset

    h = df['hour'].to_numpy(dtype=float)
    df['daylight_prev'] = ((h > sunrise) & ((h - 1.0) < sunset)).astype('int8')

    df['daylight_instant'] = ((h >= sunrise) & (h < sunset)).astype('int8')
    df['daylight_next']    = (((h + 1.0) > sunrise) & (h < sunset)).astype('int8')

    t_mid_clock = h - 0.5
    lst_mid = t_mid_clock + tc_h 
    H_deg = 15.0 * (lst_mid - 12.0)
    H = np.deg2rad(H_deg)
    elev = np.arcsin(np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H))
    df['solar_elevation'] = np.rad2deg(elev)

    return df

#endregion

""" 
위 함수들의 실행 코드
1) load_data
2) upsample_20min_linear
3) feature_engineering
4) peak_holidays
5) attach_solar_geometry
6) build_stats_features
7) test에 없는 피쳐를 제작
 """
train_path = './Energy/train.csv'
test_path = './Energy/test.csv'
building_path = './Energy/building_info.csv'
sub_path = './Energy/sample_submission.csv'
train, test, submission = load_data(train_path, test_path, building_path, sub_path)

print(f"    Before Feature Engineering - train : {train.shape} | test : {test.shape}")  # (204000, 15) (16800, 12)
train = date_features(train)
test = date_features(test)

numeric_cols= [
    'temperature','humidity', 'windspeed',
]

train = upsample_20min_linear(train, time_col="date", group_cols=["building_num"], numeric_cols=numeric_cols, minutes=AUGMENTATION)

train = feature_engineering(train)
test = feature_engineering(test)

train = peak_holidays(train)
test = peak_holidays(test, is_train=False)

KOREA_MAINLAND_LON = 127.7

train = attach_solar_geometry(train, lat=37.5665, lon=KOREA_MAINLAND_LON, tz_hours=9.0)
test  = attach_solar_geometry(test,  lat=37.5665, lon=KOREA_MAINLAND_LON, tz_hours=9.0)

train, test = build_stats_features(train, test, mode='byb')

test = test.assign(
    sunshine=test.get('sunshine', 0.0),
    solar=test.get('solar', 0.0),
    power_consumption=test.get('power_consumption', 0.0),
    is_interpolated=test.get('is_interpolated', 0.0),
)

print(f"    After Feature Engineering  - train : {train.shape} | test : {test.shape}")  # (612000, 55) (16800, 55)

#endregion

#endregion STEP 1

print("[STEP 2] INTERPOLATION") # train 데이터의 일사 보간, test 데이터의 일조 일사 예측 생성

#region STEP 2

""" 
유틸리티 함수: _to_py, _feat_sig, _cache_paths, _save_best_params, _load_best_params
<목적>
옵튜나 결과를 JSON으로 안전하게 캐싱/재사용하기 위한 함수
피처 구성이 바뀌면 키가 달라져서 다른 파라미터 재사용을 방지

요점
_to_py(o) : numpy/스칼라/리스트/딕셔너리를 JSON 직렬화 가능 타입으로 변환
_feat_sig(feature_cols) : 피처 리스트를 정렬→문자열→md5 해시 8자리로 요약(피처 서명)
_cache_paths(...) : cache_dir/optuna/{prefix}-{target}-feat{sig}.json 경로와 키 생성
_save_best_params(...) : 임시 파일(.tmp)에 쓰고 os.replace로 원자적 저장(중간 실패 방지)
_load_best_params(path) : 저장된 best_params, best_value 로드

요약
피처 구성이 바뀌면 키도 바뀌는 안전한 옵튜나 캐시
 """
def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if hasattr(o, "item"):
        try: return o.item()
        except Exception: pass
    if isinstance(o, (np.floating, np.integer)):
        return o.tolist()
    return o

def _feat_sig(feature_cols):
    s = ",".join(map(str, sorted(feature_cols)))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def _cache_paths(cache_dir, prefix, target_col, feature_cols, cache_key=None):
    key = cache_key or f"{prefix}-{target_col}-feat{_feat_sig(feature_cols)}"
    d = os.path.join(cache_dir, "optuna")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{key}.json")
    return p, key

def _save_best_params(path, key, best_params, best_value, extra=None):
    payload = {
        "key": key,
        "best_value": _to_py(best_value),
        "best_params": _to_py(best_params),
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra: payload["extra"] = _to_py(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_best_params(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params"), data.get("best_value", None)


""" 
train데이터를 확인해본 결과 일사를 기록하지 않은 건물들이 존재함. (전부 0)
그 부분을 먼저 보간한 이후 모델로 넘어가야 한다고 판단

학습 데이터에서 일사량(solar)이 0으로 잘못 기록된 건물(zero_bnos)의 주간 구간을 XGB/LGB 앙상블로 보간(impute)
zero_bnos는 SMAPE 평가에서 제외(METRIC_EXCLUDE_BNOS)
1) 학습 대상 필터링: zero_bnos 아닌 건물 + daylight_prev==1(주간)만으로 모델 학습 → 리크 방지, 물리 일관성 유지.
2) 피처 선택: drop_cols와 target_col을 제외한 숫자형만 사용.
3) 부분 샘플링(기본 30%)으로 옵튜나 탐색 속도 ↑.
4) 시간 정렬(TimeSeriesSplit): _time_order_index로 시계열 순서를 보장하고 TSS로 OOF.

모델/탐색:
XGB/LGB 하이퍼파라미터를 동시에 제안하고, fold마다 median imputing 후 학습.
검증 예측을 w_xgb * xgb + w_lgb * lgb로 가중 앙상블 후 SMAPE 최소화.
tqdm 진행바에 현재 best를 실시간 표시.
캐시 재사용: feature_cols 서명 기반 키로 best_params를 저장/재활용.
최종 모델: 각 fold의 best_iteration 평균을 사용해 n_estimators를 합리적으로 결정한 뒤, 전체 데이터로 재학습.
적용(보간): zero_bnos & 주간 위치만 예측치로 덮어쓰고, 야간은 0으로 강제.

요약:
주간만 학습·예측하는 XGB/LGB 앙상블로, 0으로 기록된 건물의 solar를 안전하게 보간
 """
def impute_solar_for_zero_bnos_cv_optuna(
    train: pd.DataFrame,
    zero_bnos,
    bno_col: str = "building_num",
    target_col: str = "solar",
    daylight_col: str = "daylight_prev",
    drop_cols=None,
    seed: int = SEED,
    n_splits: int = 5,
    n_trials: int = 20,
    sample_frac: float = 0.3,
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
):

    def _smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    METRIC_EXCLUDE_BNOS = {9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98}

    def _detect_time_col(df: pd.DataFrame):
        for c in ["date", "일시", "datetime", "timestamp", "ts", "time"]:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if s.notna().any():
                    return c, s
        return None, None

    def _time_order_index(df: pd.DataFrame) -> pd.Index:
        c, s = _detect_time_col(df)
        if c is None:
            return df.index
        return s.sort_values(kind="mergesort").index

    def _optimize_with_pbar(objective, n_trials, desc, seed):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)
        def _cb(study, trial):
            if study.best_value is not None:
                pbar.set_postfix_str(f"best={study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        pbar.write(f"    [{desc}] SMAPE: {study.best_value:.4f}")
        pbar.close()
        return study

    drop_cols = set(drop_cols or [])

    mask_train = (~train[bno_col].isin(zero_bnos)) & (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col}) and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    ord_idx = _time_order_index(df_sample)
    Xs_ord = df_sample.loc[ord_idx, feature_cols]
    ys_ord = df_sample.loc[ord_idx, target_col].astype(float)

    def objective(trial):
        xgb_params = dict(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("xgb_eta", 0.03, 0.2, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
            subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 2.0, 10.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 3.0),
            random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mae",
        )
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 2000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.03, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 3.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
        )

        tss = TimeSeriesSplit(n_splits=n_splits)
        N = len(Xs_ord)
        oof_pred_ord = np.zeros(N, dtype=float)

        for tr_idx, val_idx in tss.split(np.arange(N)):
            X_tr, X_val = Xs_ord.iloc[tr_idx], Xs_ord.iloc[val_idx]
            y_tr, y_val = ys_ord.iloc[tr_idx], ys_ord.iloc[val_idx]

            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50)
            xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

            lgb = LGBMRegressor(**lgb_params)
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )

            px = xgb.predict(X_val_f)
            pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        oof_series = pd.Series(oof_pred_ord, index=df_sample.loc[ord_idx].index)
        oof_pred = oof_series.reindex(df_sample.index).to_numpy()

        mask_metric = ~df_sample[bno_col].isin(METRIC_EXCLUDE_BNOS)
        if mask_metric.sum() == 0:
            return _smape(df_sample[target_col].to_numpy(), oof_pred)
        return _smape(df_sample.loc[mask_metric, target_col].to_numpy(),
                      oof_series.reindex(df_sample.index)[mask_metric].to_numpy())

    cache_path, cache_tag = _cache_paths(cache_dir, prefix="impute-solar", target_col=target_col,
                                         feature_cols=feature_cols, cache_key=cache_key)
    best_params = None
    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params, _best_value = _load_best_params(cache_path)
            if not all(k in best_params for k in ["xgb_n_estimators","xgb_eta","lgb_n_estimators","lgb_learning_rate"]):
                best_params = None
        except Exception:
            best_params = None

    if best_params is None:
        study = _optimize_with_pbar(objective, n_trials=n_trials, desc="optuna(solar-impute)", seed=seed)
        best_params = study.best_params
        _save_best_params(cache_path, cache_tag, best_params, study.best_value,
                          extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed})

    ord_idx_full = _time_order_index(df_train)
    X_ord = df_train.loc[ord_idx_full, feature_cols]
    y_ord = df_train.loc[ord_idx_full, target_col].astype(float)

    xgb_best = {
        "n_estimators": int(best_params["xgb_n_estimators"]),
        "learning_rate": float(best_params["xgb_eta"]),
        "max_depth": int(best_params["xgb_max_depth"]),
        "subsample": float(best_params["xgb_subsample"]),
        "colsample_bytree": float(best_params["xgb_colsample_bytree"]),
        "min_child_weight": float(best_params["xgb_min_child_weight"]),
        "reg_lambda": float(best_params["xgb_reg_lambda"]),
        "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae",
    }
    lgb_best = {
        "n_estimators": int(best_params["lgb_n_estimators"]),
        "learning_rate": float(best_params["lgb_learning_rate"]),
        "num_leaves": int(best_params["lgb_num_leaves"]),
        "max_depth": int(best_params["lgb_max_depth"]),
        "min_child_samples": int(best_params["lgb_min_child_samples"]),
        "subsample": float(best_params["lgb_subsample"]),
        "colsample_bytree": float(best_params["lgb_colsample_bytree"]),
        "reg_lambda": float(best_params["lgb_reg_lambda"]),
        "max_bin": int(best_params.get("lgb_max_bin", 255)),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    Nfull = len(X_ord)
    oof_pred_ord = np.zeros(Nfull, dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tss.split(np.arange(Nfull)):
        X_tr, X_val = X_ord.iloc[tr_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y_ord.iloc[tr_idx], y_ord.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        xgb = XGBRegressor(**xgb_best, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_series = pd.Series(oof_pred_ord, index=df_train.loc[ord_idx_full].index)
    oof_pred_full = oof_series.reindex(df_train.index).to_numpy()

    mask_metric_full = ~df_train[bno_col].isin(METRIC_EXCLUDE_BNOS).to_numpy()
    if mask_metric_full.sum() == 0:
        oof_smape_best = float(_smape(df_train[target_col].to_numpy(), oof_pred_full))
    else:
        oof_smape_best = float(_smape(df_train.loc[mask_metric_full, target_col].to_numpy(),
                                      oof_series.reindex(df_train.index)[mask_metric_full].to_numpy()))

    med_full = X_ord.median()
    X_full = X_ord.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_best["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_best, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_ord, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_ord)

    train_filled = train.copy()
    m_day = (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 1)
    if m_day.any():
        feats = train_filled.loc[m_day, feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats)
        pred = w_xgb * px + w_lgb * pl
        train_filled.loc[m_day, target_col] = np.clip(pred, 0, None)

    train_filled.loc[
        (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 0),
        target_col
    ] = 0.0

    return train_filled, oof_smape_best

""" 
목적
sunshine 또는 solar(또는 임의의 target_col)를 예측하기 위한 일반화 함수.
학습 OOF 성능(SMAPE)과 테스트 예측치를 동시에 산출.

CV/탐색/샘플링: n_splits, n_trials, sample_frac
앙상블 가중치: w_xgb, w_lgb
restrict_daylight=True: 주간만 학습/예측, 야간은 0으로 설정(sunshine은 [0,1]로 클리핑)
캐시 옵션: reuse_optuna, cache_dir, cache_key

1) 학습 데이터 구성: exclude_bnos 제거, 필요 시 daylight_prev==1만 사용.
2) 피처 선택: 숫자형만(drop_cols/target_col 제외).
3) 옵튜나 탐색: (샘플링 데이터 + 시간정렬) 위에서 XGB/LGB 동시 제안 → TSS OOF SMAPE 최소화.
4) 캐시 재사용: 피처 서명 기반 키로 best_params 저장/로드.
5) OOF 재계산: 전체 학습 데이터로 TSS를 다시 돌려 최종 OOF SMAPE 계산.
6) 최종 학습: fold별 best_iteration 평균으로 n_estimators를 결정 → 전체 학습셋으로 최종 모델 2개(XGB/LGB) 학습.

테스트 예측:
restrict_daylight=True: 주간만 예측, 야간 0.
타깃이 sunshine이면 예측을 [0,1]로 클립.
 """
def train_predict_test_target_cv_optuna(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str, 
    exclude_bnos=None,
    bno_col: str = "building_num",
    daylight_col: str = "daylight_prev",
    drop_cols=None,
    seed: int = 42,
    n_splits: int = 5,
    n_trials: int = 20, 
    sample_frac: float = 0.3,
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    restrict_daylight: bool = True,
    # ▼ 추가
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
):
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    def _detect_time_col(df: pd.DataFrame):
        for c in ["date", "일시", "datetime", "timestamp", "ts", "time"]:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if s.notna().any():
                    return c, s
        return None, None

    def _time_order_index(df: pd.DataFrame) -> pd.Index:
        c, s = _detect_time_col(df)
        if c is None:
            return df.index
        return s.sort_values(kind="mergesort").index

    def _optimize_with_pbar(objective, n_trials, desc, seed):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)
        def _cb(study, trial):
            if study.best_value is not None:
                pbar.set_postfix_str(f"best={study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        pbar.write(f"    [{desc}] SMAPE: {study.best_value:.4f}")
        pbar.close()
        return study

    drop_cols = set(drop_cols or [])
    exclude_bnos = set(exclude_bnos or [])

    # 학습 데이터
    mask_train = (~train[bno_col].isin(exclude_bnos))
    if restrict_daylight:
        mask_train &= (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 피처
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col}) and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    # 샘플 + 시간정렬
    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    ord_idx = _time_order_index(df_sample)
    Xs_ord = df_sample.loc[ord_idx, feature_cols]
    ys_ord = df_sample.loc[ord_idx, target_col].astype(float)

    def objective(trial):
        xgb_params = dict(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("xgb_eta", 0.03, 0.2, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
            subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 2.0, 10.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 3.0),
            random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mae",
        )
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 2000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.03, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 3.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
        )

        tss = TimeSeriesSplit(n_splits=n_splits)
        N = len(Xs_ord)
        oof_pred_ord = np.zeros(N, dtype=float)

        for tr_idx, val_idx in tss.split(np.arange(N)):
            X_tr, X_val = Xs_ord.iloc[tr_idx], Xs_ord.iloc[val_idx]
            y_tr, y_val = ys_ord.iloc[tr_idx], ys_ord.iloc[val_idx]

            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50)
            xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

            lgb = LGBMRegressor(**lgb_params)
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )

            px = xgb.predict(X_val_f)
            pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        oof_series = pd.Series(oof_pred_ord, index=df_sample.loc[ord_idx].index)
        oof_pred = oof_series.reindex(df_sample.index).to_numpy()
        return smape(df_sample[target_col].to_numpy(), oof_pred)

    cache_path, cache_tag = _cache_paths(cache_dir, prefix=f"predict-{target_col}",
                                         target_col=target_col, feature_cols=feature_cols,
                                         cache_key=cache_key)
    best_params = None
    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params, _best_value = _load_best_params(cache_path)
            if not all(k in best_params for k in ["xgb_n_estimators","xgb_eta","lgb_n_estimators","lgb_learning_rate"]):
                best_params = None
        except Exception:
            best_params = None

    if best_params is None:
        study = _optimize_with_pbar(objective, n_trials=n_trials, desc=f"optuna({target_col})", seed=seed)
        best_params = study.best_params
        _save_best_params(cache_path, cache_tag, best_params, study.best_value,
                          extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed})

    ord_idx_full = _time_order_index(df_train)
    X_ord = df_train.loc[ord_idx_full, feature_cols]
    y_ord = df_train.loc[ord_idx_full, target_col].astype(float)

    xgb_best = {
        "n_estimators": int(best_params["xgb_n_estimators"]),
        "learning_rate": float(best_params["xgb_eta"]),
        "max_depth": int(best_params["xgb_max_depth"]),
        "subsample": float(best_params["xgb_subsample"]),
        "colsample_bytree": float(best_params["xgb_colsample_bytree"]),
        "min_child_weight": float(best_params["xgb_min_child_weight"]),
        "reg_lambda": float(best_params["xgb_reg_lambda"]),
        "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae",
    }
    lgb_best = {
        "n_estimators": int(best_params["lgb_n_estimators"]),
        "learning_rate": float(best_params["lgb_learning_rate"]),
        "num_leaves": int(best_params["lgb_num_leaves"]),
        "max_depth": int(best_params["lgb_max_depth"]),
        "min_child_samples": int(best_params["lgb_min_child_samples"]),
        "subsample": float(best_params["lgb_subsample"]),
        "colsample_bytree": float(best_params["lgb_colsample_bytree"]),
        "reg_lambda": float(best_params["lgb_reg_lambda"]),
        "max_bin": int(best_params.get("lgb_max_bin", 255)),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    Nfull = len(X_ord)
    oof_pred_ord = np.zeros(Nfull, dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tss.split(np.arange(Nfull)):
        X_tr, X_val = X_ord.iloc[tr_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y_ord.iloc[tr_idx], y_ord.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        xgb = XGBRegressor(**xgb_best, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_series = pd.Series(oof_pred_ord, index=df_train.loc[ord_idx_full].index)
    oof_smape = float(smape(df_train[target_col].to_numpy(),
                            oof_series.reindex(df_train.index).to_numpy()))

    med_full = X_ord.median()
    X_full = X_ord.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_best["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_best, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_ord, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_ord)

    test_pred = pd.Series(index=test.index, dtype=float)
    if restrict_daylight:
        m_day = (test[daylight_col] == 1)
        feats = test.loc[m_day, feature_cols].fillna(med_full)
        if len(feats) > 0:
            px = xgb_final.predict(feats)
            pl = lgb_final.predict(feats)
            pred = w_xgb * px + w_lgb * pl
            if target_col == 'sunshine':
                test_pred.loc[m_day] = np.clip(pred, 0, 1)
            else:
                test_pred.loc[m_day] = np.clip(pred, 0, None)
        test_pred.loc[~m_day] = 0.0
    else:
        feats = test[feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats)
        pred = w_xgb * px + w_lgb * pl
        if target_col == 'sunshine':
            test_pred[:] = np.clip(pred, 0, 1)
        else:
            test_pred[:] = np.clip(pred, 0, None)

    return test_pred, oof_smape


""" 
train의 일사가 결측되어 있는 부분을 보간
 """
drop_cols = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption', 'building_num'
]

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

train_filled, smape_score = impute_solar_for_zero_bnos_cv_optuna(
    train=train,
    zero_bnos=zero_bnos,
    drop_cols=drop_cols,
    seed=50,
    n_splits=N_SPLITS,
    n_trials=N_TRIALS,
    sample_frac=SAMPLE_SIZE,
    reuse_optuna=True,
    cache_key = "train-solar-impute"
)

print("    > SMAPE :", smape_score)


""" 
test의 일조와 일사를 예측해서 생성
 """
drop_cols_for_sunshine = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'solar','power_consumption', 'building_num'
]

drop_cols_for_solar = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption','building_num'
]

test['sunshine'], oof_sunshine = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='sunshine',
    exclude_bnos=None,
    drop_cols=drop_cols_for_sunshine,
    seed=SEED, 
    n_splits=N_SPLITS, 
    n_trials=N_TRIALS, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
    reuse_optuna=True,
    cache_key = "test-sunshine-predict",
)

print("    > SMAPE :", oof_sunshine)

test['solar'], oof_solar = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='solar',
    exclude_bnos=None,
    drop_cols=drop_cols_for_solar,
    seed=701, 
    n_splits=N_SPLITS, 
    n_trials=N_TRIALS, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
    reuse_optuna=True,
    cache_key = "test-solar-predict",
)

print("    > SMAPE :", oof_solar)

#endregion STEP 2

print("\nPreprocessing Session Completed")  #STEP 2의 보간의 결과를 train과 비교하여 확인
print("Description")

#region DESCRIPTION/FEATURES
def _stats(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return dict(
        mean=np.nanmean(x),
        var=np.nanvar(x, ddof=ddof),
        std=np.nanstd(x, ddof=ddof),
    )

print("[Train - Zero_bnos]")
for i in zero_bnos:
    s = train_filled.loc[train_filled['building_num'] == i, 'solar']
    if len(s) == 0:
        print(f"Building Number: {i}  (no rows)")
        continue
    print(f"Building Number: {i}")
    print(f"Min ~ Max : {s.min():.6f} ~ {s.max():.6f}")
print(f"> SMAPE : {smape_score:.6f}")

print("\n[Test - Sunshine]")
tr_sun = train_filled['sunshine']
te_sun = test['sunshine']
st_tr  = _stats(tr_sun, ddof=1)
st_te  = _stats(te_sun, ddof=1)
print("Stats (train sunshine vs test sunshine)")
print(f"- train : mean={st_tr['mean']:.6f}, var={st_tr['var']:.6f}, std={st_tr['std']:.6f}")
print(f"- test  : mean={st_te['mean']:.6f}, var={st_te['var']:.6f}, std={st_te['std']:.6f}")
print(f"> SMAPE : {oof_sunshine:.6f}")

print("\n[Test - Solar]")
tr_sol = train_filled['solar']
te_sol = test['solar']
st_tr  = _stats(tr_sol, ddof=1)
st_te  = _stats(te_sol, ddof=1)
print("Stats (train solar vs test solar)")
print(f"- train : mean={st_tr['mean']:.6f}, var={st_tr['var']:.6f}, std={st_tr['std']:.6f}")
print(f"- test  : mean={st_te['mean']:.6f}, var={st_te['var']:.6f}, std={st_te['std']:.6f}")
print(f"> SMAPE : {oof_solar:.6f}\n")

""" 
87번 건물을 확인해본 결과 학교임에도 한낮에 전력 소비량이 최소이고 한밤중에 최대치를 보이는 것으로 확인
6월 29일 01시 기준으로 데이터의 연속성이 끊겨있다고 판단하여 그 시간 기준으로 전력 소비량을 6시간 뒤로 미루고, 그 사이 전력소비량이 nan인 부분을 drop
 """
def shift_power_from_6h_and_trim(
    df: pd.DataFrame,
    bno: int = 87,
    start_str: str = "20240629 01",
    hours: int = 6,
    id_col: str = "building_num",
    time_col: str = "date",
    time_fallback_col: str = "일시",
    target_col: str = 'power_consumption'
) -> pd.DataFrame:

    out = df.copy()

    # 0) datetime 보장
    if time_col not in out.columns:
        if time_fallback_col not in out.columns:
            raise KeyError(f"'{time_col}'도 '{time_fallback_col}'도 없습니다.")
        try:
            out[time_col] = pd.to_datetime(out[time_fallback_col].astype(str),
                                           format="%Y%m%d %H", errors="raise")
        except Exception:
            out[time_col] = pd.to_datetime(out[time_fallback_col], errors="coerce")
        if out[time_col].isna().any():
            raise ValueError(f"'{time_fallback_col}' 파싱 실패가 있습니다.")

    start_dt = pd.to_datetime(start_str, format="%Y%m%d %H")

    m_bno = (out[id_col] == bno)
    idx_bno_sorted = out[m_bno].sort_values(time_col).index

    idx_after = out.loc[idx_bno_sorted][out.loc[idx_bno_sorted, time_col] >= start_dt].index
    n_after = len(idx_after)
    if n_after == 0:
        return out
    if n_after > hours:
        src_idx  = idx_after[:-hours]
        dest_idx = idx_after[hours:]
        out.loc[dest_idx, target_col] = out.loc[src_idx, target_col].to_numpy()

    drop_idx = set()

    drop_idx.update(idx_after[:min(hours, n_after)])

    drop_idx.update(idx_after[max(0, n_after - hours):])

    if drop_idx:
        out = out.drop(index=list(drop_idx)).reset_index(drop=True)

    return out

""" 
위에서 확인한 intervals와 singles를 drop하는 함수
 """
def drop_outlier_times(df, intervals=None, singles=None, inclusive='both'):

    df = df.copy()

    s = df['일시'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    if s.str.len().eq(10).all():   # YYYYMMDDHH
        df['dt'] = pd.to_datetime(s, format='%Y%m%d%H', errors='coerce')
    else:
        df['dt'] = pd.to_datetime(df['일시'], errors='coerce', infer_datetime_format=True)

    mask = pd.Series(False, index=df.index)

    if intervals:
        for bno, spans in intervals.items():
            for start, end in spans:
                sdt = pd.to_datetime(str(start), format='%Y%m%d%H')
                edt = pd.to_datetime(str(end),   format='%Y%m%d%H')
                mask |= (df['building_num'] == bno) & df['dt'].between(sdt, edt, inclusive=inclusive)
    if singles:
        for bno, tlist in singles.items():
            for t in tlist:
                tdt = pd.to_datetime(str(t), format='%Y%m%d%H')
                mask |= (df['building_num'] == bno) & (df['dt'] == tdt)

    removed = df.loc[mask].sort_values(['building_num','dt'])
    kept    = df.loc[~mask].reset_index(drop=True)
    removed = removed.drop(['dt'], axis=1)
    kept = kept.drop(['dt'], axis=1)
    
    return kept, removed

train_filled = shift_power_from_6h_and_trim(train_filled)
train_clean, train_removed = drop_outlier_times(train_filled, intervals, singles, inclusive='both')

""" 
일조 일사를 train과 test 모두 보간 및 생성했기 때문에 일조 일사로 건물별 태양광 관려 피쳐를 생성하는 함수
하루 총 일조시간 피쳐, 일일 태양광 발전량, 시간당 태양광 발전량, 일사량 기반 일일 태양광 발전량 등 새로운 파생 피쳐를 제작
pcs 가 있는 건물은 최대 전력 저장량, pcs 방전 시간 피쳐를 생성
 """
def make_building_features(
    df: pd.DataFrame,
    date_col: str = "date",
    bno_col: str = "building_num",
    sunshine_col: str = "sunshine",
    solar_col: str = "solar",
    pvc_col: str = "pvc",
    ess_col: str = "ess",
    pcs_col: str = "pcs",
    pr: float = 0.80,
) -> pd.DataFrame:
    out = df.copy()

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["_day"] = out[date_col].dt.floor("D")
    grp_keys = [bno_col, "_day"]

    day_agg = (
        out.groupby(grp_keys, dropna=False)
           .agg(
               sunshine_day_hours=(sunshine_col, "sum"),
               solar_day_mj_m2=(solar_col, "sum"),
           )
           .reset_index()
    )
    day_agg["solar_day_kwh_m2"] = day_agg["solar_day_mj_m2"] / 3.6

    out = out.merge(day_agg, on=grp_keys, how="left")

    out["sunshine_day_hours"] = out["sunshine_day_hours"].clip(lower=0, upper=24)

    out["pv_day_kwh_est"] = (
        out[pvc_col].astype(float)
        * out["solar_day_kwh_m2"].astype(float)
        * float(pr)
    ).fillna(0).clip(lower=0)

    day_sum_solar = (
        out.groupby(grp_keys, dropna=False)[solar_col]
           .transform("sum")
    )
    num = out[solar_col].to_numpy(dtype=float)
    den = day_sum_solar.to_numpy(dtype=float)
    w = np.divide(num, den, out=np.zeros(len(out), dtype=float), where=den > 0)
    out["pv_hour_kwh_est"] = (out["pv_day_kwh_est"].to_numpy(dtype=float) * w)
    out["pv_hour_kwh_est"] = out["pv_hour_kwh_est"].fillna(0).clip(lower=0)

    out["pvc_per_day"]  = out[pvc_col].astype(float) * out["sunshine_day_hours"].astype(float)
    out["solar_by_pvc"] = out[solar_col].astype(float) * out[pvc_col].astype(float)

    if pcs_col in out.columns:
        out["pv_to_ess_kwh_cap"] = np.minimum(
            out["pv_hour_kwh_est"], out[pcs_col].astype(float).clip(lower=0)
        )
    if ess_col in out.columns:
        out["ess_hours_at_pvc"] = np.divide(
            out[ess_col].astype(float),
            np.maximum(out[pvc_col].astype(float), 1e-6)
        )

    out = out.drop(columns=["_day"])
    return out

train_clean = make_building_features(train_clean)
test = make_building_features(test)

#endregion DESCRIPTION/FEATURES

print("[STEP 3] MODEL")         # model 부분
print(f"    Train for MODEL | {train_clean.shape}")
print(f"    Test for MODEL  | {test.shape}\n")

#region STEP 3

#region MODEL FUNCTIONS

""" 
목적: 후보 컬럼 중에서 train/test에 모두 존재하고 숫자형인 것만 골라서 실제 학습 피처 리스트를 제작
이유: 컬럼 불일치/비수치형으로 인한 런타임 에러 방지 + 모델 입력 일관성 확보.
출력: 유효한 피처 컬럼 리스트.
 """
def _valid_features(train_df, test_df, cols):
    cols = [c for c in cols if (c in train_df.columns) and (c in test_df.columns)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    return cols

""" 
목적: 입력 분포를 보고 자동으로 log1p 변환이 유리한 피처를 선택

결측만 있는 컬럼은 제외.
음수가 있으면 제외(로그 불가).
분포가 롱테일이면 선택: q95/q5 > 20 또는 max > 1000.

추가적으로 아래 강제 포함 목록을 항상 로그:
['all_area','cooling_area','pvc','ess','pcs','pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day','solar_by_pvc','Facility_Density']
이유: 스케일 축소/왜도 감소로 트리 모델의 분할 안정성 및 일반화 향상.
출력: 로그 변환 대상 컬럼 리스트
 """
def auto_select_log_cols(df, cols):
    log_cols = []
    for c in cols:
        x = pd.to_numeric(df[c], errors='coerce')
        if x.isna().all():
            continue
        x = x[x.notna()]
        if x.min() < 0:
            continue
        q5, q95 = np.percentile(x, [5, 95])
        rng_ok = (q95 / (q5 + 1e-6)) > 20 or x.max() > 1_000
        if rng_ok:
            log_cols.append(c)
    forced = ['all_area','cooling_area','pvc','ess','pcs',
              'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
              'solar_by_pvc','Facility_Density']
    for c in forced:
        if c in cols and c not in log_cols:
            log_cols.append(c)
    return log_cols

""" 
목적: 선정된 features로 설계행렬 X를 만들고, log_cols에는 log1p(clip0) 변환을 적용.
세부: 음수값은 0으로 클리핑 후 np.log1p 적용 → 수치 안정화.
출력: 변환된 DataFrame X.
 """
def _make_X(df, features, log_cols):
    X = df[features].copy()
    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    return X

""" 
목적: 옵튜나에서 합쳐 저장한 파라미터(xgb_*/lgb_*)를 모델별 딕셔너리로 분리하고 키를 정규화.
정규화: eta/lr → learning_rate.
출력: (xgb_params, lgb_params).
 """
def _split_best_params(best_params: dict):
    if not isinstance(best_params, dict):
        return {}, {}
    xgb = {k[4:]: v for k, v in best_params.items() if k.startswith("xgb_")}
    lgb = {k[4:]: v for k, v in best_params.items() if k.startswith("lgb_")}
    # 키 정규화
    if "eta" in xgb: xgb["learning_rate"] = xgb.pop("eta")
    if "lr"  in xgb: xgb["learning_rate"] = xgb.pop("lr")
    if "lr"  in lgb: lgb["learning_rate"] = lgb.pop("lr")
    return xgb, lgb

""" 
model 학습기 함수

목적: 단일 타깃에 대해 TimeSeriesSplit OOF로 XGB/LGB 하이브리드 학습을 수행하고,
옵튜나로 찾은 파라미터 + fold별 best-iteration을 반영해 최종 모델 2개를 반환한다.

흐름
1) 피처 정제/로그 선택:
feat_ok(유효 숫자형) 필터 → auto_select_log_cols로 로그 대상 선별 → _make_X.

2)타깃 로그 스케일 학습:
y_all_raw = clip0(float) 후 log1p로 학습, 예측 시 expm1.
이유: 양의 연속 타깃의 롱테일/스파이크를 완화하여 분할 안정성↑.

3) CV 루프(TSS):
fold마다 중앙값 대치(median impute) → XGB/LGB 학습(early stopping 50) →
검증 예측을 expm1 후 w_xgb/w_lgb로 가중합.

4) SMAPE(사전 정의 함수 사용)를 OOF에서 계산(필요시 샘플링로 속도 조절).

5) Optuna 탐색 & 캐시:
n_trials 반복으로 SMAPE 최소 파라미터 탐색, tqdm 진행바에 best 갱신 표시.
cache_prefix-target-feat_sig.json에 best_params/best_value 저장(재사용 시 로드).

6) 최종 n_estimators 결정:
fold별 best_iteration의 평균값으로 최종 n_estimators 산정(최소 100 보장).
캐시에 저장된 xgb_final_n/lgb_final_n가 있으면 우선 사용.

7) 풀데이터 재학습:
로그 타깃으로 XGB/LGB 2개 최종 모델 학습.

반환물: xgb_final, lgb_final, med_full(중앙값 대치용), feat_ok, log_cols, oof_smape, oof_series

장점
시간 누출 방지: TSS 사용 + 중앙값 대치만 사용.
재현성/속도: 캐시 재사용 + 부분 샘플링 탐색

요약
Optuna+TSS로 XGB/LGB 하이브리드 OOF를 안정화하고, fold 평균 best-iteration으로 최종 모델을 만드는 학습기.
 """
def _train_one_model_cv_optuna(
    train_df, target_col, feature_cols, seed=42,
    n_splits=5, n_trials=30, sample_frac=0.3,
    w_xgb=0.5, w_lgb=0.5,
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
    cache_prefix: str = "tss",
):
    feat_ok = [c for c in feature_cols
               if (c in train_df.columns) and pd.api.types.is_numeric_dtype(train_df[c])]
    if not feat_ok:
        raise ValueError("유효한 feature가 없습니다.")
    log_cols = auto_select_log_cols(train_df, feat_ok)

    y_all_raw = (
        pd.to_numeric(train_df[target_col], errors='coerce')
        .fillna(0).astype(float)
        .clip(lower=0)
    )
    X_all = _make_X(train_df, feat_ok, log_cols)

    def _run_cv_once(xgb_params, lgb_params, sample_frac_for_metric=1.0, collect_best_iters=False):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_xgb = np.full(len(X_all), np.nan, dtype=float)
        oof_lgb = np.full(len(X_all), np.nan, dtype=float)
        was_val = np.zeros(len(X_all), dtype=bool)
        iters_x, iters_l = [], []

        for tr_idx, val_idx in tscv.split(X_all):
            X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            y_tr = np.log1p(y_all_raw.iloc[tr_idx].to_numpy())
            y_val = np.log1p(y_all_raw.iloc[val_idx].to_numpy())

            xgb = XGBRegressor(
                **xgb_params,
                random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mae",
                early_stopping_rounds=50
            )
            xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

            lgb = LGBMRegressor(
                **lgb_params,
                random_state=seed, n_jobs=-1
            )
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )

            px = xgb.predict(X_val_f)
            pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_xgb[val_idx] = np.expm1(px).clip(min=0)
            oof_lgb[val_idx] = np.expm1(pl).clip(min=0)
            was_val[val_idx] = True

            if collect_best_iters:
                iters_x.append(getattr(xgb, "best_iteration", None))
                iters_l.append(getattr(lgb, "best_iteration_", None))

        mask = was_val
        pred_mix = w_xgb * oof_xgb[mask] + w_lgb * oof_lgb[mask]

        idx = np.where(mask)[0]
        if sample_frac_for_metric < 1.0 and len(idx) > 200:
            k = max(200, int(len(idx) * sample_frac_for_metric))
            rng = np.random.default_rng(seed)
            sub = np.sort(rng.choice(idx, size=k, replace=False))
            sm = float(smape(y_all_raw.to_numpy()[sub],
                             (w_xgb * oof_xgb + w_lgb * oof_lgb)[sub]))
        else:
            sm = float(smape(y_all_raw.to_numpy()[mask], pred_mix))

        return sm, (w_xgb * oof_xgb + w_lgb * oof_lgb), was_val, iters_x, iters_l

    cache_path, cache_tag = _cache_paths(
        cache_dir=cache_dir,
        prefix=cache_prefix,
        target_col=target_col,
        feature_cols=feat_ok,
        cache_key=cache_key
    )

    best_params_combined = None
    best_value = None
    xgb_final_n_cached = None
    lgb_final_n_cached = None

    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params_combined, best_value = _load_best_params(cache_path)
            # extra의 final_n도 있으면 회수
            with open(cache_path, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            _extra = _payload.get("extra", {})
            xgb_final_n_cached = _extra.get("xgb_final_n", None)
            lgb_final_n_cached = _extra.get("lgb_final_n", None)
        except Exception:
            best_params_combined = None

    if best_params_combined is None:
        from optuna.samplers import TPESampler
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        pbar = tqdm(total=n_trials, desc="[optuna(hparams)]", ncols=80,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)

        def objective(trial):
            xgb_params = dict(
                lr=trial.suggest_float("xgb_lr", 0.02, 0.2, log=True),
                max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
                subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
                n_estimators=trial.suggest_int("xgb_n_estimators", 800, 4000),
            )
            lgb_params = dict(
                lr=trial.suggest_float("lgb_lr", 0.02, 0.2, log=True),
                num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
                min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
                subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("lgb_reg_alpha", 0.0, 5.0),
                max_bin=trial.suggest_int("lgb_max_bin", 127, 511),
                n_estimators=trial.suggest_int("lgb_n_estimators", 800, 4000),
                verbosity=-1,
            )
            combined = {f"xgb_{k}": v for k, v in xgb_params.items()}
            combined.update({f"lgb_{k}": v for k, v in lgb_params.items()})

            sm, *_ = _run_cv_once(
                xgb_params={**xgb_params, "learning_rate": xgb_params.pop("lr")},
                lgb_params={**lgb_params, "learning_rate": lgb_params.pop("lr")},
                sample_frac_for_metric=max(0.1, min(1.0, float(sample_frac))),
                collect_best_iters=False
            )
            try:
                pbar.set_postfix_str(f"best_smape: {min(study.best_value, sm):.6f}")
            except Exception:
                pbar.set_postfix_str("best_smape: n/a")
            # trial user_attrs에 기록(선택)
            trial.set_user_attr("combined_params", combined)
            return sm

        def _cb(study, trial):
            pbar.update(1)

        try:
            study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        finally:
            try:
                pbar.set_postfix_str(f"best_smape: {study.best_value:.6f}")
            except Exception:
                pass
            pbar.close()

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise RuntimeError("Optuna: 완료된 trial이 없습니다.")

        best_trial = study.best_trial
        best_params_combined = best_trial.user_attrs.get("combined_params", None)
        if best_params_combined is None:
            bp = study.best_params
            xgb_bp = {f"xgb_{k}": v for k, v in bp.items() if k.startswith("xgb_") or k in {
                "xgb_lr","xgb_max_depth","xgb_min_child_weight","xgb_subsample","xgb_colsample_bytree",
                "xgb_reg_lambda","xgb_reg_alpha","xgb_n_estimators"
            }}
            lgb_bp = {f"lgb_{k}": v for k, v in bp.items() if k.startswith("lgb_") or k in {
                "lgb_lr","lgb_num_leaves","lgb_min_child_samples","lgb_subsample","lgb_colsample_bytree",
                "lgb_reg_lambda","lgb_reg_alpha","lgb_max_bin","lgb_n_estimators"
            }}
            best_params_combined = {**xgb_bp, **lgb_bp}

        best_value = float(study.best_value)
        _save_best_params(
            cache_path, cache_tag, best_params_combined, best_value,
            extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed}
        )

    best_xgb_raw, best_lgb_raw = _split_best_params(best_params_combined)
    best_xgb = {
        **best_xgb_raw,
        "learning_rate": best_xgb_raw.get("learning_rate", best_xgb_raw.get("lr", 0.1)),
    }
    best_lgb = {
        **best_lgb_raw,
        "learning_rate": best_lgb_raw.get("learning_rate", best_lgb_raw.get("lr", 0.1)),
        "verbosity": -1,
    }

    oof_smape, oof_pred, was_val, itx, itl = _run_cv_once(
        best_xgb, best_lgb, sample_frac_for_metric=1.0, collect_best_iters=True
    )
    oof_series = pd.Series(oof_pred, index=X_all.index, name=f"OOF_{target_col}")

    med_full = X_all.median()
    X_full = X_all.fillna(med_full)
    y_full_log = np.log1p(y_all_raw.to_numpy())

    def _safe_iter(avg, default_min=100):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default_min
        try:
            return int(max(default_min, round(avg)))
        except Exception:
            return default_min

    xgb_final_n = _safe_iter(np.nanmean([i for i in itx if i is not None]))
    lgb_final_n = _safe_iter(np.nanmean([i for i in itl if i is not None]))
    if isinstance(xgb_final_n_cached, (int, float)) and xgb_final_n_cached > 0:
        xgb_final_n = int(xgb_final_n_cached)
    if isinstance(lgb_final_n_cached, (int, float)) and lgb_final_n_cached > 0:
        lgb_final_n = int(lgb_final_n_cached)

    xgb_final = XGBRegressor(
        **{**best_xgb, "n_estimators": int(xgb_final_n),
           "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae"}
    )
    xgb_final.fit(X_full, y_full_log, verbose=False)

    lgb_final = LGBMRegressor(
        **{**best_lgb, "n_estimators": int(lgb_final_n),
           "random_state": seed, "n_jobs": -1}
    )
    lgb_final.fit(X_full, y_full_log)

    try:
        _save_best_params(
            cache_path, cache_tag, best_params_combined, best_value,
            extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed,
                   "xgb_final_n": int(xgb_final_n), "lgb_final_n": int(lgb_final_n)}
        )
    except Exception:
        pass

    return (
        xgb_final, lgb_final, med_full, feat_ok, log_cols,
        float(oof_smape), oof_series
    )

""" 
model 학습 후 추론기
목적: 학습기에서 반환한 모델 팩으로 일괄 추론을 수행.
입력: models_pack = (xgb_final, lgb_final, med, feats, log_cols)
과정: _make_X → 결측을 med로 대치 → 모델 예측 → expm1 복원 → 0.5/0.5 단순 가중 평균.
옵션: clip0=True면 음수는 0으로 클립.
출력: numpy.ndarray 예측값.

비고: 학습 시 사용했던 w_xgb/w_lgb와 달리, 여기서는 0.5/0.5 고정.
필요하면 이 함수를 w_xgb, w_lgb 인자를 받도록 확장할 수 있음

요약
학습 시와 동일한 변환을 적용해 XGB/LGB 예측을 복원·평균해 최종 값을 만든다.
 """
def _predict_with_models(df, models_pack, clip0=True):
    xgb_final, lgb_final, med, feats, log_cols = models_pack
    X = _make_X(df, feats, log_cols).fillna(med)
    px = xgb_final.predict(X)
    pl = lgb_final.predict(X)
    pred = 0.5 * np.expm1(px) + 0.5 * np.expm1(pl)
    if clip0:
        pred = np.clip(pred, 0, None)
    return pred

""" 
model에서 사용할 피쳐들을 종류별로 나눠서 먼저 정의해둠
모델은 항상 태양광만 있는 건물, 태양광과 ess가 있는 건물, 둘 다 없는 건물로 나눠서 학습하기 때문에 피쳐를 관련 된 것만 선택
 """
pvc_features = [
    'building_type','groupID','holidays', 'is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'all_area','cooling_area',
    'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc',
    'PVC_per_CA','Facility_Density',
]

ess_features = [
    'building_type','groupID','holidays', 'is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'all_area','cooling_area',
    'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc','pv_to_ess_kwh_cap','ess_hours_at_pvc',
    'PVC_per_CA','Facility_Density',
]

no_pvc_features = [
    'building_type','groupID','holidays','is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar',
    'all_area','cooling_area',
]

# -------------------- 1) 세그먼트 모델 (PVC/ESS/No-PVC) --------------------
""" 
1) segmented model

목적
ESS(PCS 포함)/PVC-only/No-PVC 세그먼트로 학습 데이터를 나눠 패턴 이질성을 반영하고, 각 세그먼트에 특화된 피처 집합으로 모델을 학습

전역 피처 리스트: ess_features, pvc_features, no_pvc_features (각각에 building_num 추가하여 사용)

흐름
세그먼트 분할
ESS: pvc>0 & pcs>0
PVC: pvc>0 & pcs==0
NoPVC: pvc==0
(train/test 각각에서 존재 여부 확인, test에 해당 세그먼트가 있을 때만 학습/예측 실행)

학습 & 예측
세그먼트별로 _train_one_model_cv_optuna 호출 → XGB/LGB 하이브리드 + TSS OOF
예측은 _predict_with_models로 수행 (학습 시와 동일 전처리/로그 스케일 복원)

결과 수집
preds: 테스트 전 구간 예측을 세그먼트별로 채워 합산
oofs: 세그먼트별 OOF SMAPE ({'ESS':…, 'PVC':…, 'NoPVC':…})
oof_series_full: 학습 인덱스와 동일한 길이의 OOF 시리즈(세그먼트별로 위치 채움)

요약
ESS/PVC/No-PVC로 나눠 각기 다른 피처·하이퍼로 학습하는 세그먼트 특화 모델

 """
def model_segmented(train_clean, test, target='power_consumption', seed=SEED, n_splits=5, n_trials=20, sample_size=0.3):
    print("  [model_segmented] activate")
    pvc_pos_train = (train_clean.get('pvc', 0).fillna(0) > 0)
    pcs_pos_train = (train_clean.get('pcs', 0).fillna(0) > 0)
    seg_ess_tr  = pvc_pos_train & pcs_pos_train
    seg_pvc_tr  = pvc_pos_train & (~pcs_pos_train)
    seg_none_tr = ~pvc_pos_train

    pvc_pos_test = (test.get('pvc', 0).fillna(0) > 0)
    pcs_pos_test = (test.get('pcs', 0).fillna(0) > 0)
    seg_ess_te  = pvc_pos_test & pcs_pos_test
    seg_pvc_te  = pvc_pos_test & (~pcs_pos_test)
    seg_none_te = ~pvc_pos_test

    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    preds = pd.Series(0.0, index=test.index)
    oofs  = {}
    
    pvc_for_seg = pvc_features.copy()
    pvc_for_seg.append('building_num')
    
    ess_for_seg = ess_features.copy()
    ess_for_seg.append('building_num')
    
    no_pvc_for_seg = no_pvc_features.copy()
    no_pvc_for_seg.append('building_num')

    # ESS
    tr_ess = train_clean.loc[seg_ess_tr]
    if len(tr_ess) > 50 and seg_ess_te.any():
        print(f"    [PCS Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_ess, target, _valid_features(train_clean, test, ess_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-ESS", cache_prefix="seg"
        )
        preds.loc[seg_ess_te] = _predict_with_models(test.loc[seg_ess_te], (xgb,lgb,med,feats,logs))
        oofs['ESS'] = oof
        oof_series_full.loc[tr_ess.index] = oof_series
        print(f"    > SMAPE : {oof}")

    # PVC only
    tr_pvc = train_clean.loc[seg_pvc_tr]
    if len(tr_pvc) > 50 and seg_pvc_te.any():
        print(f"    [PV Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_pvc, target, _valid_features(train_clean, test, pvc_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-PVC", cache_prefix="seg"
        )
        preds.loc[seg_pvc_te] = _predict_with_models(test.loc[seg_pvc_te], (xgb,lgb,med,feats,logs))
        oofs['PVC'] = oof
        oof_series_full.loc[tr_pvc.index] = oof_series 
        print(f"    > SMAPE : {oof}")

    # No-PVC
    tr_none = train_clean.loc[seg_none_tr]
    if len(tr_none) > 50 and seg_none_te.any():
        print(f"    [No-PV Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_none, target, _valid_features(train_clean, test, no_pvc_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-NoPVC", cache_prefix="seg"
        )
        preds.loc[seg_none_te] = _predict_with_models(test.loc[seg_none_te], (xgb,lgb,med,feats,logs))
        oofs['NoPVC'] = oof
        oof_series_full.loc[tr_none.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return preds.values, oofs, oof_series_full 

""" 
2) by type model

목적
building_type 별로 모델을 따로 학습해, 유형 고유의 부하 패턴을 반영

피처: ess_features + ['building_num']를 포괄적 피처로 사용(유형 간 공통성 최대화)

흐름
테스트에 등장하는 building_type 값 기준으로 루프
유형별 학습셋(tr_t)이 50행 미만이면 스킵
_train_one_model_cv_optuna로 학습 → _predict_with_models로 해당 유형 구간 예측
type_scores[t] = OOF SMAPE, oof_series_full에 OOF 배치

요약
건물 유형 단위로 모델을 분리해 유형별 패턴을 학습하는 전략.
 """
def model_by_type(train_clean, test, target='power_consumption', type_col='building_type', seed=SEED, n_splits=5, n_trials=20, sample_size=0.3):
    print("  [model_by_type] activate")
    pred = pd.Series(0.0, index=test.index)
    ess_for_type = ess_features.copy()
    ess_for_type.append('building_num')
    type_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    for t in sorted(test[type_col].dropna().unique()):
        print(f"    [Building Type] {t}")
        tr_t = train_clean[train_clean[type_col] == t]
        te_t = test[test[type_col] == t]
        if len(tr_t) < 50:
            continue
        feats = _valid_features(train_clean, test, ess_for_type)  # 포괄적 피처
        xgb,lgb,med,fs,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_t, target, feats,
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key=f"type-{t}", cache_prefix="type"
        )
        pred.loc[te_t.index] = _predict_with_models(te_t, (xgb,lgb,med,fs,logs))
        type_scores[str(t)] = oof
        oof_series_full.loc[tr_t.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return pred.values, type_scores, oof_series_full

""" 
3) by building_numbers model

목적
건물번호 단위(per-building)로 완전 분리 학습하여, 개별 건물의 미시적 패턴/설비 특성을 최대한 흡수

피처 풀: ess_features 있으면 사용, 없으면 PVC 피처로 폴백

흐름
테스트에 등장하는 건물번호만 대상으로 루프
학습 데이터가 30행 미만이면 스킵
건물 단위로 세그먼트 판정
ESS(pvc>0 & pcs>0) → ess_features
PVC(pvc>0 & pcs==0) → pvc_features
NoPVC(pvc==0) → no_pvc_features
_valid_features로 실사용 피처 확정 → _train_one_model_cv_optuna로 학습
_predict_with_models로 해당 건물 구간 예측, OOF를 oof_series_full에 배치
각 건물의 OOF SMAPE를 b_scores[str(b)]로 기록

요약
건물별·설비상태별로 최적 피처로 학습해, 개별 건물의 고유 패턴을 최대한 반영.
 """
def model_by_building_num(train_clean, test, target='power_consumption', bno_col='building_num', pvc_col='pvc', pcs_col='pcs', seed=SEED, n_splits=5, n_trials=20, sample_size=0.7):
    print("  [model_by_building_num] activate (per-building, ESS/PVC/NoPVC aware)")

    pred = pd.Series(0.0, index=test.index, dtype=float)
    b_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index, dtype=float)
    try:
        _ess_features = ess_features
    except NameError:
        _ess_features = None

    for b in sorted(test[bno_col].dropna().unique()):
        tr_b = train_clean[train_clean[bno_col] == b]
        te_b = test[test[bno_col] == b]
        if len(tr_b) < 30:
            print(f"    [Building {b}] skip (train rows < 30)")
            continue

        has_pvc_train = (pvc_col in tr_b.columns) and (tr_b[pvc_col].fillna(0) > 0).any()
        has_pvc_test  = (pvc_col in te_b.columns) and (te_b[pvc_col].fillna(0) > 0).any()
        has_pvc = bool(has_pvc_train or has_pvc_test)

        has_pcs_train = (pcs_col in tr_b.columns) and (tr_b[pcs_col].fillna(0) > 0).any()
        has_pcs_test  = (pcs_col in te_b.columns) and (te_b[pcs_col].fillna(0) > 0).any()
        has_pcs = bool(has_pcs_train or has_pcs_test)

        if has_pvc and has_pcs:
            seg = 'ESS'
            feat_list = _ess_features if _ess_features is not None else pvc_features
        elif has_pvc:
            seg = 'PVC'
            feat_list = pvc_features
        else:
            seg = 'NoPVC'
            feat_list = no_pvc_features

        feats = _valid_features(train_clean, test, feat_list)
        print(f"    [Building {b}] segment={seg} | features={len(feats)}")

        if len(feats) == 0:
            print(f"      > skip: no valid features for building {b}")
            continue

        xgb, lgb, med, fs, logs, oof, oof_series = _train_one_model_cv_optuna(
            tr_b, target, feats,
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key=f"bno-{b}-{seg}", cache_prefix="bno"
        )
        pred.loc[te_b.index] = _predict_with_models(te_b, (xgb, lgb, med, fs, logs))
        oof_series_full.loc[tr_b.index] = oof_series
        b_scores[str(b)] = oof
        print(f"      > SMAPE : {oof:.6f}")

    return pred.values, b_scores, oof_series_full

#endregion MODEL FUNCTIONS

""" 
모델 실행 부분

1) M1 세그먼트(ESS/PVC/NoPVC) 모델, M2 건물유형별 모델, M4 건물번호별 모델을 각각 학습·예측
2) 각 모델의 OOF SMAPE 평균을 콘솔에 출력
3) 테스트 예측치를 동일 가중(1/3,1/3,1/3) 으로 앙상블 → pred_final
4) 학습 구간 OOF들도 행 단위로 정규화 가중 합하여 최종 OOF SMAPE 계산
5) submission['answer']에 최종 예측을 기록

요약
세그먼트·유형·건물별 세 모델을 동일 가중으로 앙상블하고, 행별 정규화 OOF로 최종 SMAPE를 계산해 리더보드 제출값을 만듦
 """
target = 'power_consumption'

print("[M1] Segmented (ESS/PVC/NoPVC)")
pred1, oof1, oof1_s = model_segmented(
    train_clean, test, target=target, seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_MODEL
)
avg1 = sum(oof1.values()) / len(oof1)
print(f"[Segmented Model] SMAPE : {avg1}\n")

print("[M2] By Type")
pred2, oof2, oof2_s = model_by_type(
    train_clean, test, target=target, type_col='building_type', seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_MODEL
)
avg2 = sum(oof2.values()) / len(oof2)
print(f"[By Type Model] SMAPE : {avg2}\n")

print("[M4] By BuildingNum (ESS/PVC/NoPVC AWARE)")
pred4, oof4, oof4_s = model_by_building_num(
    train_clean, test, target=target, bno_col='building_num', seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_BNUM
)
avg4 = sum(oof4.values()) / len(oof4)
print(f"[By BuildingNum Model] SMAPE : {avg4}\n")

#endregion STEP 3

print("[STEP 4] SAVE")          # 앙상블 및 저장 부분

#region STEP 4
W1, W2, W4 = 1/3, 1/3, 1/3
pred_final = W1*pred1 + W2*pred2 + W4*pred4

oof_df = pd.DataFrame({
    'm1': oof1_s,   # 세그먼트 OOF
    'm2': oof2_s,   # 유형별 OOF
    'm4': oof4_s,   # 번호별 OOF
})

w = np.array([W1, W2, W4], dtype=float)
W = pd.DataFrame(np.broadcast_to(w, oof_df.shape), index=oof_df.index, columns=oof_df.columns)

W = W.where(oof_df.notna(), 0.0)
wsum = W.sum(axis=1)
mask = (wsum > 0) & oof_df.notna().any(axis=1)

W_norm = W.div(wsum, axis=0).where(mask, 0.0)
oof_ens = (oof_df.fillna(0.0) * W_norm).sum(axis=1)

final_oof_smape = smape(
    train_clean.loc[mask, 'power_consumption'].to_numpy(),
    oof_ens.loc[mask].to_numpy()
)
print(f"[OOF] Final Ensemble SMAPE: {final_oof_smape}\n")

submission['answer'] = pred_final

today = datetime.datetime.now().strftime('%Y%m%d')
score_str = ("nan" if np.isnan(final_oof_smape) else f"{final_oof_smape:.4f}").replace('.', '_')
filename = f"{SEED}_SMAPE_{score_str}_{version}.csv"
submission.to_csv(save_path + filename, index=False)
    
#endregion STEP 4

print(f"[{version}] COMPLETED")
