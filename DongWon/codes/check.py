import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

client = OpenAI(api_key=api_key)
# 계정·모델 접근만 확인용: 에러 없이 실행되면 OK
models = client.models.list()
print(len(models.data))