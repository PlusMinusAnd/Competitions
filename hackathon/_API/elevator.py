# 코드 재시작으로 파일 재로드
import pandas as pd

# 파일 경로
file_path = "./hackathon/data/서울시 지하철역 엘리베이터 위치정보.csv"

# 파일 로드 (cp949 인코딩)
elevator_df = pd.read_csv(file_path, encoding="cp949")

# 첫 번째 열 미리 보기
elevator_df.iloc[:, 0].head()

# 경도, 위도 분리
elevator_df['longitude'] = elevator_df['노드 WKT'].str.extract(r'POINT\(([^ ]+)')[0].astype(float)
elevator_df['latitude'] = elevator_df['노드 WKT'].str.extract(r'POINT\([^ ]+ ([^)]+)\)')[0].astype(float)

# 저장
output_path = "./hackathon/data/서울시_지하철역_엘리베이터_위치정보_분리.csv"
elevator_df.to_csv(output_path, index=False, encoding='utf-8-sig')

output_path
