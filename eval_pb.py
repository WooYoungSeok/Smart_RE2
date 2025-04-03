import os
import pandas as pd
import datetime
from utils import get_parse_fn
import numpy as np
import ast
import re

# -------------------------------
# 설정 및 데이터 불러오기
# -------------------------------
predictions_file = "predictions_gpt-4o-mini_svamp_20250402_165105.xlsx"  # 예측 파일 경로
df = pd.read_excel(predictions_file)
# 실행 코드: python eval_pb.py

def parse_platinum_target(s):
    # 대괄호를 제거한 후, 작은따옴표 안의 내용을 추출합니다.
    return np.array(re.findall(r"'(.*?)'", s))

# platinum_target 열의 각 행을 변환합니다.
df["platinum_target"] = df["platinum_target"].apply(lambda x: parse_platinum_target(x) if isinstance(x, str) else x)

# 평가할 태스크 목록 (예측 파일에 f'{task}_pred' 열이 존재해야 함)
tasks_list = ['platinum_prompt', 'platinum_prompt_no_cot',
              'RE2', 'sRE2', 'RE2_no_cot', 'sRE2_no_cot'] # 'sum', 'table', 'graph', 'bullet_point', 'sum_no_cot', 'table_no_cot', 'graph_no_cot', 'bullet_point_no_cot', 'sRE2_m1', 'sRE2_m2', 'sRE2_m1_no_cot', 'sRE2_m2_no_cot'
               

# -------------------------------
# 데이터셋에 따른 파싱 전략 선택
# -------------------------------
# 예측 파일 이름에서 데이터셋명을 추출 (예: "winograd")
dataset_name = predictions_file.split('_')[2].lower()

# 데이터셋별 파싱 전략 매핑
parsing_strategy_mapping = {
    "singleop": "math",
    "singleeq": "math",
    "multiarith": "math",
    "gsm8k": "math",
    "svamp": "math",
    "singleq": "math",
    "mmlu": "multiple_choice",  # mmlu math
    "winograd": "multiple_choice",  # winograd schema challenge
    "bbh": "bbh_multiple_choice",
    "drop": "text",
    "hotpotqa": "text",
    "squad": "squad",
    "logic": "bbh_multiple_choice",
    "navigate": "text",
    "object": "text",
    "tab": "text"
}

# 데이터셋명에 맞는 파싱 전략 선택, 없으면 "unknown"으로 설정
parsing_strategy = parsing_strategy_mapping.get(dataset_name, "unkown")
print(f"Dataset: {dataset_name}, Parsing Strategy: {parsing_strategy}")

# utils.py의 get_parse_fn을 사용하여 파싱 함수 생성
parse_fn = get_parse_fn(parsing_strategy)

# -------------------------------
# 각 태스크별로 정답 문자열 추출
# -------------------------------
for task in tasks_list:
    # f"{task}_pred" 열의 값을 대상으로 파싱 함수를 적용하여 정답 문자열 추출
    df[f"{task}_extracted"] = df[f"{task}_pred"].apply(lambda x: parse_fn(x) if isinstance(x, str) else "")

# -------------------------------
# 간단한 평가 예시 (정확도 계산)
# -------------------------------
# 추출한 정답 문자열과 ground truth("platinum_target")를 비교하여 정확도(문자열 일치 여부) 계산

evaluation = {}
for task in tasks_list:
    accuracy = df.apply(lambda row:
        any(row[f"{task}_extracted"].lower().strip() == str(t).strip().lower()
            for t in row["platinum_target"]), axis=1).mean()  
    evaluation[task] = accuracy

print("Task Accuracies:")
for task, acc in evaluation.items():
    print(f"{task}: {acc}")

# -------------------------------
# 최종 출력 파일 구성 및 저장
# -------------------------------
output_columns = ["platinum_prompt_no_cot", "platinum_target"] + [f"{task}_extracted" for task in tasks_list]
output_df = df[output_columns]

model_name = predictions_file.split('_')[1]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"evaluation_results_{model_name}_{dataset_name}_{timestamp}.xlsx"
output_df.to_excel(output_file, index=False)
print("Evaluation results file saved as:", output_file)

# 실행 코드: python eval_pb.py