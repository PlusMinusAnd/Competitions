import re
import os
import pandas as pd
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

torch.cuda.set_per_process_memory_fraction(0.875, device=0)
test = pd.read_csv('./Finance/test.csv')

# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options

# Few-shot 예시 정의
FEW_SHOT_EXAMPLES = {
    "multiple_choice": [
        {
            "question": "다음 중 HTTP와 HTTPS의 가장 주요한 차이점은 무엇인가요?",
            "options": [
                "1 전송 속도의 차이",
                "2 암호화 여부",
                "3 데이터 형식의 차이",
                "4 포트 번호만 다름"
            ],
            "answer": "2"
        },
        {
            "question": "데이터베이스에서 ACID 속성에 해당하지 않는 것은?",
            "options": [
                "1 원자성(Atomicity)",
                "2 일관성(Consistency)",
                "3 격리성(Isolation)",
                "4 보안성(Security)"
            ],
            "answer": "4"
        },
        {
            "question": "다음 중 객체지향 프로그래밍의 4대 특징이 아닌 것은?",
            "options": [
                "1 캡슐화(Encapsulation)",
                "2 상속성(Inheritance)",
                "3 다형성(Polymorphism)",
                "4 컴파일성(Compilation)"
            ],
            "answer": "4"
        }
    ],
    "subjective": [
        {
            "question": "클라우드 컴퓨팅의 주요 장점 3가지를 설명하세요.",
            "answer": "1) 비용 효율성: 초기 하드웨어 투자 없이 필요한 만큼만 사용하여 비용 절감 2) 확장성: 트래픽 증가 시 즉시 리소스 확장 가능 3) 접근성: 인터넷만 있으면 언제 어디서나 접근 가능"
        },
        {
            "question": "API와 SDK의 차이점을 간단히 설명하세요.",
            "answer": "API는 소프트웨어 간 상호작용을 위한 인터페이스 규약이며, SDK는 특정 플랫폼이나 서비스 개발을 위한 도구와 라이브러리 모음입니다. API는 '무엇을 할 수 있는지' 정의하고, SDK는 '어떻게 구현할지' 도구를 제공합니다."
        },
        {
            "question": "Git에서 merge와 rebase의 차이점을 설명하세요.",
            "answer": "Merge는 두 브랜치를 합치면서 새로운 merge commit을 생성하여 브랜치 히스토리를 유지합니다. Rebase는 한 브랜치의 커밋들을 다른 브랜치 위로 재배치하여 선형적인 히스토리를 만듭니다. Merge는 히스토리 보존, Rebase는 깔끔한 히스토리 유지에 장점이 있습니다."
        }
    ]
}

# 키워드 매칭 함수
def detect_question_type(question_text):
    """
    질문 텍스트에서 키워드를 찾아 전문가 타입을 결정
    """
    finance_keywords = ["금융", "투자", "신용", "예금", "대출"]
    security_keywords = ["보안", "해킹", "악성코드", "암호화", "방화벽"]
    law_keywords = ["항", "법", "벌칙", "처벌", "조항"]
    regulation_keywords = ["규제", "감독", "허가", "승인", "제한"]
    
    if any(keyword in question_text for keyword in finance_keywords):
        return "finance"
    elif any(keyword in question_text for keyword in security_keywords):
        return "security"
    elif any(keyword in question_text for keyword in law_keywords):
        return "law"
    elif any(keyword in question_text for keyword in regulation_keywords):
        return "regulation"
    else:
        return "general"

# Few-shot 프롬프트 생성기
def make_few_shot_prompt(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        
        # 객관식 Few-shot 예시들
        examples_text = ""
        for i, example in enumerate(FEW_SHOT_EXAMPLES["multiple_choice"]):
            examples_text += f"예시 {i+1}:\n"
            examples_text += f"질문: {example['question']}\n"
            examples_text += "선택지:\n"
            examples_text += f"{chr(10).join(example['options'])}\n"
            examples_text += f"답변: {example['answer']}\n\n"
        
        prompt = (
            "당신은 금융 전문가입니다.\n"
            "아래 예시들을 참고하여 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"{examples_text}"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        # 주관식 Few-shot 예시들
        examples_text = ""
        for i, example in enumerate(FEW_SHOT_EXAMPLES["subjective"]):
            examples_text += f"예시 {i+1}:\n"
            examples_text += f"질문: {example['question']}\n"
            examples_text += f"답변: {example['answer']}\n\n"
        
        prompt = (
            "당신은 금융 전문가입니다.\n"
            "아래 예시들을 참고하여 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
            f"{examples_text}"
            f"질문: {text}\n\n"
            "답변:"
        )   
    return prompt

# 기존 프롬프트 생성기 (키워드 기반 전문가 선택)
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        question_type = detect_question_type(question)
        
        if question_type == "finance":
            prompt = (
                "당신은 금융분석가 입니다.\n"
                "금융시장과 투자상품에 대한 깊은 이해를 가진 금융분석가로서 답변해 주세요.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
            )
        elif question_type == "security":
            prompt = (
                "당신은 사이버보안 전문가 입니다.\n"
                "사이버 위협 분석과 방어 기술에 정통한 보안 전문가로서 답변해 주세요.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
            )
        elif question_type == "law":
            prompt = (
                "당신은 법률전문가 입니다.\n"
                "해당 분야 법률 조항과 판례에 정통한 법률전문가로서 답변해 주세요.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
            )
        elif question_type == "regulation":
            prompt = (
                "당신은 규제정책 분석가 입니다.\n"
                "산업 규제와 감독 절차를 전문적으로 분석하는 규제정책 분석가로서 답변해 주세요.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"                
            )
        else:
            prompt = (
                "당신은 금융 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
            )
    else:
        question_type = detect_question_type(text)
        
        if question_type == "finance":
            prompt = (
                "당신은 금융분석가 입니다.\n"
                "금융시장과 투자상품에 대한 깊은 이해를 가진 금융분석가로서 답변해 주세요.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
        elif question_type == "security":
            prompt = (
                "당신은 사이버보안 전문가 입니다.\n"
                "사이버 위협 분석과 방어 기술에 정통한 보안 전문가로서 답변해 주세요.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
        elif question_type == "law":
            prompt = (
                "당신은 법률전문가 입니다.\n"
                "해당 분야 법률 조항과 판례에 정통한 법률전문가로서 답변해 주세요.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
        elif question_type == "regulation":
            prompt = (
                "당신은 규제정책 분석가 입니다.\n"
                "산업 규제와 감독 절차를 전문적으로 분석하는 규제정책 분석가로서 답변해 주세요.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"             
            )
        else:
            prompt = (
                "당신은 금융 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
    
    return prompt

model_name = "beomi/gemma-ko-7b"

# Tokenizer 및 모델 로드 (4bit)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "30GiB"},
    load_in_4bit=True,
    torch_dtype=torch.float32
)

# Inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# 후처리 함수
def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    
    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text
    
preds = []

# Few-shot 프롬프트 사용
for q in tqdm(test['Question'], desc="Few-shot Inference"):
    prompt = make_few_shot_prompt(q)  # Few-shot 프롬프트 사용
    output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
    pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
    preds.append(pred_answer)
    
sample_submission = pd.read_csv('./Finance/sample_submission.csv')
sample_submission['Answer'] = preds
sample_submission.to_csv('./Finance/few_shot_submission.csv', index=False, encoding='utf-8-sig')

print("Few-shot 추론 완료! 결과가 few_shot_submission.csv로 저장되었습니다.")