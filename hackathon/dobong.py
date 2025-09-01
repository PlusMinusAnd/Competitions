import pandas as pd

# ✅ 파일 경로
dobong_path = "./hackathon/data/dobong.csv"
subway_path = "./hackathon/data/subway.csv"
elevator_path = "./hackathon/data/subway_elevator.csv"
population_path = "./hackathon/data/population.csv"

# ✅ 데이터 불러오기
dobong_df = pd.read_csv(dobong_path, encoding='cp949')
subway_df = pd.read_csv(subway_path, encoding='utf-8')
elevator_df = pd.read_csv(elevator_path, encoding='cp949')
population_df = pd.read_csv(population_path, encoding="cp949")

# ✅ 주변 개수 계산 함수
def count_nearby(df_target, lat, lon, delta=0.005):
    return df_target[
        (df_target['latitude'].between(lat - delta, lat + delta)) &
        (df_target['longitude'].between(lon - delta, lon + delta))
    ].shape[0]

# ✅ 지하철 및 엘리베이터 개수 계산
subway_counts = []
elevator_counts = []

for _, row in dobong_df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    subway_counts.append(count_nearby(subway_df, lat, lon))
    elevator_counts.append(count_nearby(elevator_df, lat, lon))

# ✅ 인구 데이터 정제 (중복 제거만!)
population_filtered = population_df.drop(columns=["지역코드.1"]).drop_duplicates(subset="지역코드")

# ✅ 병합 (region_code ↔ 지역코드 매칭)
merged_df = pd.merge(dobong_df, population_filtered, left_on="region_code", right_on="지역코드", how="left")

# ✅ 필요한 컬럼으로 정리
final_df = pd.DataFrame({
    "region_code": merged_df["region_code"],
    "lat": merged_df["latitude"],
    "lon": merged_df["longitude"],
    "subway": subway_counts,
    "sub_elevator": elevator_counts,
    "population": merged_df["소계_인구"],
    "elderly_ratio": merged_df["고령자_인구비율"],
    "disability_ratio": merged_df["전체_장애인구비율"]
})

# ✅ 저장
output_path = "./hackathon/data/dobong_final.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_path}")
