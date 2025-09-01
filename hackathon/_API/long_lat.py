import requests
import pandas as pd
import time

# 🔑 Kakao REST API 키
KAKAO_API_KEY = '5c3db37c6508aa742dde52c8e94563e4'

# 📄 변환할 주소 리스트 (예시로 직접 넣거나, CSV 불러오기)
address_list = 여기에 채워넣으시면 돼요

# 📦 결과 저장 리스트
results = []

# 📍 주소 → 좌표 변환 반복
for address in address_list:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if 'documents' in data and len(data['documents']) > 0:
        doc = data['documents'][0]
        results.append({
            "address": address,
            "road_address": doc.get("road_address", {}).get("address_name", ""),
            "lat": doc['y'],
            "lon": doc['x']
        })
    else:
        results.append({
            "address": address,
            "road_address": "",
            "lat": "",
            "lon": ""
        })
    time.sleep(0.2)  # 너무 빠른 요청 방지

# 📄 결과 저장
df = pd.DataFrame(results)
df.to_csv("address_to_coords.csv", index=False, encoding='utf-8-sig')
print(f"{len(df)}개 주소 처리 완료!")