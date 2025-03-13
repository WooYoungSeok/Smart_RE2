import os
import pandas as pd
import aiohttp
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
import datetime

# -------------------------------
# 설정 및 데이터 불러오기
# -------------------------------
# 평가할 예측 파일 이름 (실제 파일 이름에 맞게 수정)
predictions_file = "predictions_gpt-4o-mini_drop_20250313_103208.xlsx"

# 예측 파일 로드
df = pd.read_excel(predictions_file)

# platinum_target 열은 원본 그대로 사용 (특별한 처리 없이)
# 만약 platinum_target이 문자열이면 그대로 두고, array나 리스트인 경우도 그대로 유지

# 평가할 태스크 목록 (예측 파일에 f'{task}_pred' 열이 존재해야 함)
tasks_list = ['platinum_prompt', 'platinum_prompt_no_cot', 'RE2', 'sum', 'table', 'graph', 'bullet_point', 
              'sRE2', 'RE2_no_cot', 'sum_no_cot', 'table_no_cot', 
              'graph_no_cot', 'bullet_point_no_cot', 'sRE2_no_cot']

# -------------------------------
# GPT‑4o를 통한 평가 API 호출 함수 (비동기)
# -------------------------------
async def evaluate_prediction(session, ground_truth, prediction, retries=3, delay=1, model="gpt-4o", max_tokens=10, temperature=0):
    """
    GPT‑4o 모델에게 ground truth와 prediction을 입력하여, 예측이 정답인지 오답인지를 평가합니다.
    프롬프트:
      "Evaluate the following prediction against the ground truth.
       Ground truth: <ground_truth>
       Prediction: <prediction>
       Answer with 'Yes' if the prediction is correct, or 'No' if it is incorrect."
    """
    url = "https://api.openai.com/v1/chat/completions"
    my_api_key = "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"
    headers = {
        "Authorization": f"Bearer {my_api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = "You are an objective evaluator."
    prompt = (
        f"Evaluate the following prediction against the ground truth.\n"
        f"Ground truth: {ground_truth}\n"
        f"Prediction: {prediction}\n"
        "Answer with '1' if any answer of the ground truth is included in the prediction, or '0' if it is not."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {error_text}")
                result = await resp.json()
                answer = result["choices"][0]["message"]["content"].strip()
                return answer  # Expecting "Yes" or "No"
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} - Error evaluating prediction: {e}")
            await asyncio.sleep(delay)
    print("All retry attempts failed for evaluating prediction.")
    return "No"  # 기본적으로 "No" 처리

# -------------------------------
# 각 태스크별 평가 처리 함수 (비동기)
# -------------------------------
async def process_evaluation_for_task(session, df, task, semaphore):
    """
    각 행의 f"{task}_pred" 값과 platinum_target을 가지고 GPT‑4o를 호출하여 평가합니다.
    각 호출의 결과(예: "Yes" 또는 "No")와 인덱스를 반환합니다.
    """
    tasks_coroutines = []
    for idx, row in df.iterrows():
        ground_truth = row["platinum_target"]
        prediction = row[f"{task}_pred"]
        async def wrapper(idx, ground_truth, prediction):
            async with semaphore:
                eval_result = await evaluate_prediction(session, ground_truth, prediction)
                return idx, eval_result
        tasks_coroutines.append(wrapper(idx, ground_truth, prediction))
    
    results = []
    for future in tqdm_asyncio.as_completed(tasks_coroutines, total=len(tasks_coroutines), desc=f"Evaluating for {task}"):
        idx, eval_result = await future
        results.append((idx, eval_result))
    results.sort(key=lambda x: x[0])
    eval_results = [res for idx, res in results]
    return eval_results

# -------------------------------
# 메인 평가 프로세스
# -------------------------------
async def main():
    CONCURRENCY_COUNT = 10  # 동시 요청 수 조절
    semaphore = asyncio.Semaphore(CONCURRENCY_COUNT)
    async with aiohttp.ClientSession() as session:
        # 각 태스크별로 평가 수행: f"{task}_eval" 열에 결과("Yes"/"No") 저장
        for task in tasks_list:
            print(f"Processing evaluation for task: {task}")
            eval_results = await process_evaluation_for_task(session, df, task, semaphore)
            df[f"{task}_eval"] = eval_results
            print(f"Completed evaluation for task: {task}")
            await asyncio.sleep(1)
    
    # 각 태스크별 정확도 계산 (GPT‑4o 평가 결과에서 "Yes" 비율)
    evaluation = {}
    for task in tasks_list:
        # "Yes"면 True, 그렇지 않으면 False
        df[f"{task}_correct"] = df[f"{task}_eval"].apply(lambda x: True if x.strip().lower() == "yes" else False)
        evaluation[task] = df[f"{task}_correct"].mean()
    
    print("Task Accuracies:")
    for task, acc in evaluation.items():
        print(f"{task}: {acc}")
    
    # 최종 출력 파일 구성:
    # - platinum_prompt_no_cot 열 (원본 그대로)
    # - platinum_target 열 (원본 그대로)
    # - 각 태스크별 f"{task}_eval" 열 (GPT‑4o 평가 결과)
    output_columns = ["platinum_prompt_no_cot", "platinum_target"] + [f"{task}_eval" for task in tasks_list]
    output_df = df[output_columns]
    
    model_name = "gpt-4o-mini"
    # 예: predictions_file = "predictions_gpt-4o-mini_winograd_20250313_155205.xlsx"
    # dataset_name는 split('_')[2], 여기서는 "winograd"
    dataset_name = predictions_file.split('_')[2]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{model_name}_{dataset_name}_{timestamp}.xlsx"
    output_df.to_excel(output_file, index=False)
    print("Evaluation results file saved as:", output_file)

if __name__ == "__main__":
    asyncio.run(main())
