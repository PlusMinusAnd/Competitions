import requests
import pandas as pd
import time

# 🔑 Kakao REST API 키
KAKAO_API_KEY = '5c3db37c6508aa742dde52c8e94563e4'

# 📍 검색할 지역명과 키워드
# region = "전라북도 무주군"
region = '서초롯데캐슬84아파트'
keyword = "버스정류장"

# 📦 결과 저장용
results = []

# 1~45까지 최대 페이지 반복 (Kakao는 한 페이지에 15개, 최대 45개)
for page in range(1, 46):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "query": f"{region} {keyword}",
        "page": page,
        "size": 15,
        "radius": 500
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if 'documents' not in data or len(data['documents']) == 0:
        break

    for place in data['documents']:
        results.append({
            "place_name": place['place_name'],
            "address": place['road_address_name'] or place['address_name'],
            "lat": place['y'],
            "lon": place['x'],
            "category": place['category_name']
        })

    time.sleep(0.2)  # 너무 빠른 요청 방지

# 📄 결과를 CSV로 저장
df = pd.DataFrame(results)
df.to_csv("muju_hospitals.csv", index=False, encoding='utf-8-sig')
print(f"{len(df)}개 결과 저장 완료!")
