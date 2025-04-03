import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_asyncio
import time
import datetime
import os

# 실행 코드: python 4o-mini_inference.py
# 비동기 동시 요청 개수를 설정 (원하는 값으로 조정)
CONCURRENCY_COUNT = 20

# API 키 설정
my_api_key = ""

# 데이터 불러오기 (parquet 파일 예시)
data_path = "sRE2_adaptive_datasets\winograd_wsc_m.parquet"
df = pd.read_parquet(data_path)
# 'cleaning_status'가 "rejected"인 행 제거
df = df[df["cleaning_status"] != "rejected"].reset_index(drop=True)

# 파일 경로에서 데이터셋 이름 추출 (예: "gsm8k_add_task.parquet" -> "gsm8k")
basename = os.path.splitext(os.path.basename(data_path))[0]  # "gsm8k_add_task"
dataset_name = basename.split("_")[0]  # "gsm8k"

# 예측할 태스크 목록
tasks_list = ['platinum_prompt', 'platinum_prompt_no_cot',
              'RE2', 'sRE2', 'RE2_no_cot', 'sRE2_no_cot'] # 'sum', 'table', 'graph', 'bullet_point', 'sum_no_cot', 'table_no_cot', 'graph_no_cot', 'bullet_point_no_cot', 'sRE2_m1', 'sRE2_m2', 'sRE2_m1_no_cot', 'sRE2_m2_no_cot'

# 모델 이름과 타임스탬프 설정 (파일명에 반영)
model_name = "gpt-4o-mini"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
predictions_file = f"predictions_{model_name}_{dataset_name}_{timestamp}.xlsx"

async def fetch_prediction(semaphore, session, prompt, task, idx, 
                           model=model_name, max_tokens=2054, temperature=0, 
                           retries=3, delay=1):
    """
    비동기 API 호출 함수.
    semaphore로 동시 호출 수를 제한하고, 지정된 횟수(retries)만큼 재시도합니다.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {my_api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = "You are a helpful assistant."
    full_prompt = f"{prompt}\n\n[Task: {task}]"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    resp_json = await response.json()
                    pred = resp_json["choices"][0]["message"]["content"].strip()
                    return idx, pred
            except Exception as e:
                print(f"Attempt {attempt+1}/{retries} - Error in row {idx} for task {task}: {e}")
                await asyncio.sleep(delay)
        print(f"All retry attempts failed for row {idx} task {task}")
        return idx, ""

async def process_task_for_all_rows(session, df, task, semaphore):
    """
    한 태스크에 대해 DataFrame의 모든 행에 대해 비동기 호출을 수행합니다.
    각 요청의 결과와 인덱스를 함께 반환한 후, 원래 순서대로 정렬합니다.
    """
    tasks_coroutines = [
        fetch_prediction(semaphore, session, row["platinum_prompt"], task, idx)
        for idx, row in df.iterrows()
    ]
    results = []
    for future in tqdm_asyncio.as_completed(tasks_coroutines, total=len(tasks_coroutines), desc=f"Task {task}"):
        idx, pred = await future
        results.append((idx, pred))
    results.sort(key=lambda x: x[0])
    predictions = [pred for idx, pred in results]
    return predictions

async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY_COUNT)
    async with aiohttp.ClientSession() as session:
        # 각 태스크별로 예측을 수행
        for task in tasks_list:
            print(f"Processing task: {task}")
            predictions = await process_task_for_all_rows(session, df, task, semaphore)
            df[task + "_pred"] = predictions
            print(f"Completed task: {task}")
            # 태스크 간 간단한 딜레이 (필요 시)
            await asyncio.sleep(1)
    
    # 예측값만 저장 (평가 코드는 별도 파일에서 진행)
    df.to_excel(predictions_file, index=False)
    print(f"Predictions saved to '{predictions_file}'.")

if __name__ == "__main__":
    asyncio.run(main())
