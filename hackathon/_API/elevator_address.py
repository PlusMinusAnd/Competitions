import pandas as pd
import requests
import time

# ✅ 카카오 REST API 키 (본인 키로 교체)
KAKAO_REST_API_KEY = "KakaoAK 527e4b29f68cc550674a0b27c335dd27"
HEADERS = {"Authorization": KAKAO_REST_API_KEY}

# ✅ CSV 불러오기 (열 이름이 정확히 'latitude', 'longitude'라고 가정)
df = pd.read_csv("hackathon/data/final_data/elevator.csv", encoding="cp949")  # 인코딩은 필요에 따라 'utf-8'로 변경

# ✅ 좌표 → 자치구_행정동 추출 함수
def get_jachi_dong(lat, lng):
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    params = {"x": lng, "y": lat}
    try:
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        for doc in res.json().get("documents", []):
            if doc["region_type"] == "H":  # 법정동 기준
                return f"{doc['region_2depth_name']}_{doc['region_3depth_name']}"
    except Exception as e:
        print(f"Error at ({lat}, {lng}): {e}")
    return None

# ✅ 변환 적용
df["자치구_행정동"] = df.apply(lambda row: get_jachi_dong(row["latitude"], row["longitude"]), axis=1)
    # API 제한 고려
    # time.sleep(0.2)

# ✅ 결과 저장
df.to_csv("./hackathon/data/elevator_with_jachigu_dong.csv", index=False, encoding="utf-8-sig")
print("✅ 저장 완료 → elevator_with_jachigu_dong.csv")
