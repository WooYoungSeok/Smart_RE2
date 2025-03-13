import os
import ast
import pandas as pd
import aiohttp
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
import datetime

# -------------------------------
# 설정 및 데이터 불러오기
# -------------------------------
# 평가할 예측 파일 이름 (실제 파일 이름에 맞게 수정)
predictions_file = "predictions_gpt-4o-mini_winograd_20250313_155205.xlsx"

# 예측 파일 로드
df = pd.read_excel(predictions_file)

# 평가할 태스크 목록 (예측 파일에 저장된 모든 task 예측값 열)
tasks_list = ['platinum_prompt', 'platinum_prompt_no_cot', 'RE2', 'sum', 'table', 'graph', 'bullet_point', 
              'sRE2', 'RE2_no_cot', 'sum_no_cot', 'table_no_cot', 
              'graph_no_cot', 'bullet_point_no_cot', 'sRE2_no_cot']

# -------------------------------
# 최종 정답 추출 함수 (비동기)
# -------------------------------
async def extract_final_answer(session, response_text, retries=3, delay=1, model="gpt-4o", max_tokens=50, temperature=0):
    """
    chain-of-thought가 포함된 응답(response_text)에서 최종 정답만 추출합니다.
    프롬프트는 "Extract the final answer from the following response: ..."로 구성됩니다.
    """
    url = "https://api.openai.com/v1/chat/completions"
    # API 키는 실제 환경에 맞게 설정하거나 환경 변수로 불러오세요.
    my_api_key = "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"
    headers = {
        "Authorization": f"Bearer {my_api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = "You are a helpful assistant."
    prompt = f"Extract the final answer from the following response. Provide only the final answer with no additional explanation:\n\n\"{response_text}\""
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
                final_answer = result["choices"][0]["message"]["content"].strip()
                return final_answer
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} - Error extracting final answer: {e}")
            await asyncio.sleep(delay)
    print("All retry attempts failed for extracting final answer.")
    return None

# -------------------------------
# 각 태스크별 최종 정답 추출 처리 (비동기)
# -------------------------------
async def process_final_answers_for_task(session, df, task, semaphore):
    """
    각 행의 <task>_pred 응답에서 최종 정답을 추출하는 비동기 작업을 수행합니다.
    """
    tasks_coroutines = []
    for idx, row in df.iterrows():
        response_text = row[f"{task}_pred"]
        async def wrapper(idx, response_text):
            async with semaphore:
                final_ans = await extract_final_answer(session, response_text)
                return idx, final_ans
        tasks_coroutines.append(wrapper(idx, response_text))
    
    results = []
    for future in tqdm_asyncio.as_completed(tasks_coroutines, total=len(tasks_coroutines), desc=f"Extracting final for {task}"):
        idx, final_ans = await future
        results.append((idx, final_ans))
    results.sort(key=lambda x: x[0])
    final_answers = [ans for idx, ans in results]
    return final_answers

# -------------------------------
# 메인 평가 프로세스
# -------------------------------
async def main():
    CONCURRENCY_COUNT = 10  # 동시 요청 수 조절
    semaphore = asyncio.Semaphore(CONCURRENCY_COUNT)
    async with aiohttp.ClientSession() as session:
        # 각 태스크별로 chain-of-thought 응답에서 최종 정답 추출
        for task in tasks_list:
            print(f"Processing final extraction for task: {task}")
            final_answers = await process_final_answers_for_task(session, df, task, semaphore)
            df[task + "_final"] = final_answers
            print(f"Completed extraction for task: {task}")
            await asyncio.sleep(1)
    
    # -------------------------------
    # 각 태스크별 정확도 계산 및 터미널 출력
    # -------------------------------
    evaluation = {}
    for task in tasks_list:
        def check_answer(row):
            final = row[task + "_final"]
            targets = row["platinum_target"]
            if final is None:
                return False
            if isinstance(targets, (list, tuple)):
                return any(final.strip().lower() == str(t).strip().lower() for t in targets)
            return False
        df[task + "_correct"] = df.apply(check_answer, axis=1)
        evaluation[task] = df[task + "_correct"].mean()
    
    print("Task Accuracies:")
    for task, acc in evaluation.items():
        print(f"{task}: {acc:.2%}")
    
    # -------------------------------
    # 최종 출력 파일 구성: platinum_prompt_no_cot, platinum_target, 각 task의 최종 예측 값
    # -------------------------------
    output_columns = ["platinum_prompt_no_cot", "platinum_target"] + [f"{task}_final" for task in tasks_list]
    output_df = df[output_columns]
    
    model_name = "gpt-4o-mini"
    dataset_name = predictions_file.split('_')[2]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_extracted_{model_name}_{dataset_name}_{timestamp}.xlsx"
    output_df.to_excel(output_file, index=False)
    print("Evaluation extracted file saved as:", output_file)

if __name__ == "__main__":
    asyncio.run(main())
