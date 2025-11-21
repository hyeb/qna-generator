import os
import data_loader
import pandas as pd
import json

from google import genai
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field, ConfigDict


## get gemini api
load_dotenv()
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print('gemini api not available')


## load lecture context
FOLDER_PATH='data/hum/huma/C00257'

CONTEXT = data_loader.get_text_contents(FOLDER_PATH)

# SAMPLE_CONTEXT = """
# 10년 넘게 발전해 온 대화형 인공지능(AI)이 여전히 사용자 대신 실제 업무를 안정적으로 수행하는 문제를 해결하지 못한 가운데, 미국 스타트업 AUI(Augmented Intelligence)가 새로운 해법을 제시했다.

# AUI는 3일(현지시간) AI가 대화를 넘어서 실제로 행동할 수 있도록 설계된 새로운 기반 모델 ‘아폴로-1(Apollo-1)’을 공개했다.

# 현재 일부 기업 고객과 미리보기 테스트 중이며, 다음 달 정식 출시가 예고돼 있다.

# 이는 '상태 기반 신경-기호 추론(stateful neuro-symbolic reasoning)' 구조를 도입, AI가 지시된 작업을 신뢰성 있게 수행하도록 설계됐다.

# '챗GPT'나 '제미나이' '클로드' 같은 트랜스포머(Transformers) 아키텍처 기반의 대형언어모델(LLM)이 대화와 창의적 응답에는 강하지만, 업무 정확성에서는 한계를 보인다는 점을 극복하려는 시도다.

# AI의 실행 성능을 측정하는 대표 벤치마크 '터미널-벤치 하드(Terminal-Bench Hard)'에서 최신 AI 모델들의 평균 점수는 30%대에 불과하다. 항공권 예약 정확도를 평가하는 'TAU-벤치 에어라인'에서도 최고 성능을 보인 '클로드 3.7 소네트'조차 56%의 성공률에 그쳐, 절반 가까운 작업에서 실패했다.

# AUI의 상태 기반 신경-기호 추론은 트랜스포머가 사용하는 확률적 예측 구조 대신, 기호(symbolic) 논리와 신경망(neural network)을 결합해 정확한 행동을 보장하는 하이브리드 구조다.

# 기존 트랜스포머 모델은 ‘다음에 올 단어’를 예측한다. 반면, 아폴로-1은 ‘다음에 취할 행동(next action)’을 예측한다. 이를 위해 AUI는 ‘기호 상태(symbolic state)’를 기반으로 하는 구조를 도입했다.

# 신경-기호란 두가지 AI 패러다임을 결합한 것이다. 기호 계층은 의도(intent), 개체(entity), 매개변수(parameter) 같은 구조를 이해하고, 신경망 계층은 언어 유창성을 제공한다. 두 계층 사이에서 추론을 수행하는 것이 바로 아폴로-1의 핵심 두뇌다.

# 아폴로-1은 단순히 문장을 만들어내는 기존 방식과 달리, ‘폐쇄형 추론 루프(closed reasoning loop)’라는 구조로 작동한다.
# """



## Question Generation
client = genai.Client(api_key=GEMINI_API_KEY)

prompt = f""" 
당신은 기말고사를 준비하고 있는 대학생입니다. 아래 'CONTENTS'를 읽고 제대로 학습했는지 확인하기 위한 질문을 만들어야합니다.
질문은 난이도에 따라 상중하로 나뉘며, 각각 2개씩 만들어주세요.

반드시 CONTENTS 안에서 확인 가능한 내용으로 제목 생성없이 '질문'만을 생성해야합니다.
출력은 딕셔너리 형태로 제공하세요.

[CONTENTS]
{CONTEXT}
"""

model_name = 'gemini-2.5-flash-lite'

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
)

response_text = response.text

### 모델응답을 pydantic으로 파싱하여 json 구조로 변환
class DifficultyQuestions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True # python에서 low/mid/high 쓰기 위함
    )

    low : List[str] = Field(alias="하")
    mid: List[str] = Field(alias="중")
    high: List[str] = Field(alias="상")


json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
diff_questions = DifficultyQuestions.model_validate_json(json_str)

diff_df = pd.DataFrame(diff_questions)

diff_df.to_csv("questions_with_difficulty_3.csv", index=False, encoding="utf-8-sig")

print('Question Generation Success')


## Question Evaluation
JSON_STRUCT = """
```json
{
  "context_relevance": {
    "score": 0 or 1,
    "question_based_on_context": true/false
  },
  "question_importance": {
    "score": 0 or 1,
    "core_concept_related": true/false,
    "not_trivial_detail": true/false,
    "misconception_exposure": true/false
  },
  "difficulty_consistency": {
    "score": 0 or 1,
    "model_difficulty": "low" | "mid" | "high",
    "match": true/false
  },
  "question_clarity": {
    "score": 0 or 1,
    "grammatically_clear": true/false,
    "intent_clear": true/false
  },
}
""" # 출력 형태 고정


EVAL_PROMPT = f"""
당신은 대학 가의 기반 학습용 'question'을 평가하는 전문가입니다.
주어진 context(강의 내용)와 question, model_difficulty(질문 난이도)를 보고 아래 4가지 지표에 따라 질문을 평가하세요.

각 지표는 0 또는 1의 바이너리로 평가합니다.

[평가지표 1] 문맥 관련성
다음 조건이 모두 True면 1점, 하나라도 False이면 0점입니다.
- question이 context에서 다루는 개념, 내용과 관련이 있는가?
- context에서 언급되지 않은 정보나 주제가 포함되어 있지 않는가?

[평가지표 2] 질문 중요도
다음 기준 중 2개 이상이 True면 1점, 아니라면 0점입니다.
- 강의의 핵심 개념과 관련이 있는가?
- 단순한 내용의 디테일이 아니라 개념/이해 수준을 점검하는 질문인가?
- 학습자의 이해 부족 혹은 잘못 알고있는 부분을 드러내는 데 도움이 되는 질문인가?

[평가지표 3] 난이도 일관성
먼저, question을 보고 실제 난이도를 'low', 'mid', 'high' 중 하나로 판단하고,
판단 결과가 model_difficulty와 같으면 1점, 아니라면 0점입니다.

[평가지표 4] 문장 완성도
다음 기준을 모두 만족하면 1점, 아니면 0점입니다.
- 문장이 크게 어색하지 않고, 한국어 문법상 이해 가능한가?
- 질문의 의도가 한번에 이해되며, 무엇을 묻는지 분명한가?


[출력 형식]
아래의 JSON 형식으로 출력하세요. 추가 텍스트나 설명은 절대 쓰지 마세요.
{JSON_STRUCT}


[평가 대상]
context : {{CONTEXT}}
question : {{QUESTION}}
model_difficulty : {{DIFFICULTY}}
"""

eval_file_path = "questions_with_difficulty_3.csv"
eval_df = pd.read_csv(eval_file_path)
eval_df.columns = ['difficulty', 'question']


def get_prompt(context, q, diff):
    return (
        EVAL_PROMPT
        .replace("{{CONTEXT}}", context)
        .replace("{{QUESTION}}", q)
        .replace("{{DIFFICULTY}}", diff)
    )

eval_results = []

for _, row in eval_df.iterrows():
    q = row['question']
    diff = row['difficulty']
    
    prompt = get_prompt(CONTEXT, q, diff)

    e_reponse = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    raw = e_reponse.text
    eval_json_str = raw[raw.find('{'):raw.rfind('}')+1]


    try:
        result = json.loads(eval_json_str)
    except json.JSONDecodeError:
        result = {
            'parse_error': True,
            'raw': eval_json_str
        }
    
    eval_results.append(result)


evaluation_result = []

for e in eval_results:
    if "parse_error" in e:
        evaluation_result.append({
            "context_score": None,
            "importance_score": None,
            "difficulty_score": None,
            "clarity_score": None,
            "parse_error": True,
            "raw": e['raw']
        })
        continue

    cr = e.get("context_relevance", {})
    qi = e.get("question_importance", {})
    dc = e.get("difficulty_consistency", {})
    qc = e.get("question_clarity", {})

    evaluation_result.append({
        "context_score": cr.get("score"),
        "importance_score": qi.get("score"),
        "difficulty_score": dc.get("score"),
        "model_difficulty": dc.get("model_difficulty"),
        "difficulty_match": dc.get("match"),
        "clarity_score": qc.get("score"),
        "parse_error": False,
    })

eval_result_df = pd.DataFrame(evaluation_result)

final_df = pd.concat(
[eval_df.reset_index(drop=True), eval_result_df.reset_index(drop=True)],
axis=1
)

final_df.to_csv('questions_with_eval.csv', index=False, encoding='utf-8')
print('저장완료')