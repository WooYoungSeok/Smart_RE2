import os
import ast
import pandas as pd

# 평가할 evaluation_extracted 파일 이름 (실제 파일 이름으로 수정)
evaluation_extracted_file = "evaluation_extracted_gpt-4o-mini_gsm8k_20250313_103208.xlsx"

# 파일 로드
df = pd.read_excel(evaluation_extracted_file)

# platinum_target 컬럼이 문자열이면, 리스트(또는 array)로 변환
def parse_target(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            print(f"Error parsing target: {x} - {e}")
            return x
    return x

df['platinum_target'] = df['platinum_target'].apply(parse_target)

# 평가할 태스크 목록 (evaluation_extracted 파일에 각 태스크의 최종 예측값은 <task>_final 열로 저장되어 있다고 가정)
tasks_list = ['RE2', 'sum', 'table', 'graph', 'bullet_point', 
              'sRE2', 'RE2_no_cot', 'sum_no_cot', 'table_no_cot', 
              'graph_no_cot', 'bullet_point_no_cot', 'sRE2_no_cot']

# 각 행에서, 특정 태스크의 최종 예측값이 platinum_target에 포함되는지 확인하는 함수
def check_answer(row, task):
    final = row.get(f"{task}_final")
    targets = row["platinum_target"]
    if final is None:
        return False
    if isinstance(targets, (list, tuple)):
        # 대소문자 무시, 공백 제거하여 비교
        return any(final.strip().lower() == str(t).strip().lower() for t in targets)
    return False

# 각 태스크별 정확도 계산
accuracy_results = {}
for task in tasks_list:
    df[f"{task}_correct"] = df.apply(lambda row: check_answer(row, task), axis=1)
    accuracy = df[f"{task}_correct"].mean()
    accuracy_results[task] = accuracy

# 터미널에 태스크별 정확도 출력
print("Task Accuracies:")
for task, acc in accuracy_results.items():
    print(f"{task}: {acc:.2%}")

# # 정확도 결과를 Excel 파일로 저장 (모델, 데이터셋, 타임스탬프 반영)
# timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
# model_name = "gpt-4o-mini"
# dataset_name = "gsm8k"  # 예시; 필요시 파일명에서 추출 가능
# output_file = f"evaluation_accuracy_{model_name}_{dataset_name}_{timestamp}.xlsx"
# acc_df = pd.DataFrame(list(accuracy_results.items()), columns=["Task", "Accuracy"])
# acc_df.to_excel(output_file, index=False)
# print(f"Accuracy metrics saved to {output_file}")
