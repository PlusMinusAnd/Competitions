import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time

# ✅ 서울시 OpenAPI 키
API_KEY = "64504359756b696d3939554e677772"  # 예: "64504359756b696d3939554e677772"

# ✅ 정류소 ID CSV 파일 불러오기 (열 이름: 'ars_id')
df_ids = pd.read_csv("정류소ID목록.csv")  # 예: ars_id 열 포함

# ✅ 결과 저장용 리스트
results = []

# ✅ ARS ID 하나씩 API 요청
for ars_id in df_ids["ars_id"].astype(str):
    url = f"http://ws.bus.go.kr/api/rest/stationinfo/getRouteByStation?serviceKey={API_KEY}&arsId={ars_id}"
    try:
        res = requests.get(url)
        root = ET.fromstring(res.content)
        
        for item in root.iter("itemList"):
            route_name = item.find("busRouteNm").text
            route_type = item.find("routeType").text
            results.append({
                "ars_id": ars_id,
                "bus_route_nm": route_name,
                "route_type": route_type
            })
        time.sleep(0.2)  # 과도한 요청 방지
    except Exception as e:
        print(f"[오류] ars_id: {ars_id} → {e}")

# ✅ DataFrame으로 정리 후 저장
df_result = pd.DataFrame(results)
df_result.to_csv("정류소별_노선정보.csv", index=False, encoding="utf-8-sig")
print("✅ 저장 완료: 정류소별_노선정보.csv")
