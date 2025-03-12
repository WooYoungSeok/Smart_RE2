import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_asyncio
import time

# 비동기 동시 요청 개수를 설정 (원하는 값으로 조정)
CONCURRENCY_COUNT = 20

# API 키 설정
my_api_key = "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"

# 데이터 불러오기 (엑셀 파일 예시)
df = pd.read_parquet("sRE2_datasets/gsm8k_add_task.parquet")
# 'cleaning_status'가 "rejected"인 행 제거
df = df[df["cleaning_status"] != "rejected"].reset_index(drop=True)

# 예측할 태스크 목록
tasks_list = ['RE2', 'sum', 'table', 'graph', 'bullet_point', 
              'sRE2', 'RE2_no_cot', 'sum_no_cot', 'table_no_cot', 
              'graph_no_cot', 'bullet_point_no_cot', 'sRE2_no_cot']

async def fetch_prediction(semaphore, session, prompt, task, idx, 
                           model="gpt-4o-mini", max_tokens=100, temperature=0, 
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
    # 각 행에 대한 비동기 작업(task) 생성
    tasks_coroutines = [
        fetch_prediction(semaphore, session, row["platinum_prompt"], task, idx)
        for idx, row in df.iterrows()
    ]
    results = []
    # tqdm_asyncio.as_completed를 사용해 진행 상황 표시
    for future in tqdm_asyncio.as_completed(tasks_coroutines, total=len(tasks_coroutines), desc=f"Task {task}"):
        idx, pred = await future
        results.append((idx, pred))
    # 인덱스 순서대로 정렬
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
    
    # 성능 평가: 각 태스크별로 'Answer' 컬럼과 예측값의 일치 여부 (단순 exact match 예시)
    for task in tasks_list:
        df[task + '_correct'] = df.apply(lambda row: row[task + '_pred'] in row['platinum_target'], axis=1)
    
    # 태스크별 정확도 계산
    accuracy = {task: df[task + "_correct"].mean() for task in tasks_list}
    acc_df = pd.DataFrame(list(accuracy.items()), columns=["Task", "Accuracy"])
    
    # 결과 엑셀 파일로 저장
    df.to_excel("predictions_async.xlsx", index=False)
    acc_df.to_excel("evaluation_metrics_async.xlsx", index=False)
    print("Predictions and evaluations saved to 'predictions_async.xlsx' and 'evaluation_metrics_async.xlsx'.")

if __name__ == "__main__":
    asyncio.run(main())