import pandas as pd
import numpy as np
dobong = pd.read_csv("./hackathon/dobong_final.csv", encoding="utf-8-sig")
seocho = pd.read_csv("./hackathon/seocho_final.csv", encoding="utf-8-sig")

# 지역 구분 컬럼 추가
dobong["district"] = "dobong"
seocho["district"] = "seocho"

# 병합
combined = pd.concat([dobong, seocho], ignore_index=True)

# 접근성 및 MII 계산
combined["accessibility"] = combined["subway"] + combined["sub_elevator"]
epsilon = 1e-6
combined["MII"] = (combined["elderly_ratio"] + combined["disability_ratio"]) / (combined["accessibility"] + epsilon)

# 로그 변환된 MII
combined["MII_log"] = np.log1p(combined["MII"])

# 도봉/서초 각각 나눠 저장
dobong_final = combined[combined["district"] == "dobong"].drop(columns=["district"])
seocho_final = combined[combined["district"] == "seocho"].drop(columns=["district"])

# 저장
dobong_path = "./hackathon/dobong_final_with_logMII.csv"
seocho_path = "./hackathon/seocho_final_with_logMII.csv"
dobong_final.to_csv(dobong_path, index=False, encoding="utf-8-sig")
seocho_final.to_csv(seocho_path, index=False, encoding="utf-8-sig")