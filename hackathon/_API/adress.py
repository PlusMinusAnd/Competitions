import pandas as pd
import requests
import time
from tqdm import tqdm  # 진행바

# ✅ 카카오 REST API 키 입력
KAKAO_REST_API_KEY = "KakaoAK 527e4b29f68cc550674a0b27c335dd27"

# ✅ 위경도 CSV 파일 불러오기
df = pd.read_csv("./hackathon/data/subway.csv", encoding='cp949')

# ✅ 좌표 → 자치구_행정동 변환 함수
def get_dong_name(lat, lng):
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    headers = {"Authorization": KAKAO_REST_API_KEY}
    params = {"x": lng, "y": lat}
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        for doc in res.json().get("documents", []):
            if doc['region_type'] == 'H':  # 행정동
                return f"{doc['region_2depth_name']}_{doc['region_3depth_name']}"
    except Exception as e:
        print(f"⚠️ 예외 발생: {e}")
        return None

# ✅ tqdm으로 진행률 표시 및 API 호출 속도 제한
dong_names = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    name = get_dong_name(row['latitude'], row['longitude'])
    dong_names.append(name)
    time.sleep(0.2)  # 카카오 초당 10건 제한

df["행정주소"] = dong_names

# ✅ 결과 저장
save_path = "./hackathon/data/subway_re.csv"
df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {save_path}")