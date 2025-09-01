import pandas as pd
import numpy as np
import requests
import time
import requests

def get_coordinates(address, api_key):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if data.get("documents"):
        doc = data["documents"][0]
        x = doc["x"]  # 경도
        y = doc["y"]  # 위도
        return float(y), float(x)
    else:
        return None, None

# 사용 예
KAKAO_API_KEY = "527e4b29f68cc550674a0b27c335dd27"
address = "서울시 서초구 사임당로17길 90"

center_lat, center_lon = get_coordinates(address, KAKAO_API_KEY)
print(f"중심 좌표 - 위도: {center_lat}, 경도: {center_lon}")

# 2) Kakao API 키 입력
KAKAO_API_KEY = '발급받은_키_입력'
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 3) Haversine 거리 계산 함수 (미터 단위)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구 반경(m)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 4) 받은 버스정류장 CSV 불러오기
encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']

for enc in encodings:
    try:
        bus_stops = pd.read_csv("./hackathon/data/bus_stops.csv", encoding=enc)
        print(f"성공! 인코딩: {enc}")
        break
    except Exception as e:
        print(f"인코딩 {enc} 실패: {e}")
        
# 5) 위도, 경도 컬럼명은 데이터에 따라 다를 수 있으니 확인 필요
# 예시는 '위도', '경도'로 가정
bus_stops['distance_m'] = bus_stops.apply(
    lambda row: haversine(center_lat, center_lon, float(row['위도']), float(row['경도'])), axis=1)

# 6) 반경 500m 이내 필터링
bus_stops_500m = bus_stops[bus_stops['distance_m'] <= 500].copy()

print(f"반경 500m 이내 버스정류장 개수: {len(bus_stops_500m)}")

# 7) 카카오 API로 주소 및 추가 정보 보정 함수
def kakao_search_place(name, x=None, y=None):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {"query": name}
    if x and y:
        params.update({"x": x, "y": y})
    res = requests.get(url, headers=headers, params=params).json()
    docs = res.get("documents")
    if docs:
        return docs[0]
    return None

bus_stops_500m.to_csv("./hackathon/data/lotte84/롯데캐슬84_반경500m_버스정류장.csv", index=False, encoding='utf-8-sig')

print("완료! CSV 저장했어요.")
