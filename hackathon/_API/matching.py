import pandas as pd

# ✅ 파일 경로 설정
subway_path = "./hackathon/data/subway_re.csv"
region_path = "./hackathon/data/region_code.csv"
output_path = "./hackathon/data/subway_with_region_code.csv"

# ✅ CSV 파일 불러오기
try:
    subway_df = pd.read_csv(subway_path, encoding='utf-8-sig')  # 행정주소 포함된 파일
    print(f"📥 subway 데이터 로드 완료: {subway_df.shape}")
except UnicodeDecodeError:
    subway_df = pd.read_csv(subway_path, encoding='utf-8')
    print(f"📥 subway 데이터 (utf-8) 로드 완료: {subway_df.shape}")

try:
    region_df = pd.read_csv(region_path, encoding='cp949')  # region_code.csv는 주로 cp949
    print(f"📥 region_code 데이터 로드 완료: {region_df.shape}")
except UnicodeDecodeError:
    region_df = pd.read_csv(region_path, encoding='utf-8')
    print(f"📥 region_code 데이터 (utf-8) 로드 완료: {region_df.shape}")

# ✅ 병합 (행정주소 기준)
merged_df = pd.merge(
    subway_df,
    region_df,
    left_on='행정주소',
    right_on='자치구_행정동',
    how='left'
)

# ✅ 병합 후 정리
merged_df.drop(columns=['자치구_행정동'], inplace=True)

# ✅ 결과 저장
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_path}")
