import os
import ast
import pandas as pd
import aiohttp
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
from sklearn.metrics import accuracy_score
import datetime

# -------------------------------
# 설정 및 데이터 불러오기
# -------------------------------
# 평가할 예측 파일 이름 (실제 파일 이름에 맞게 수정)
predictions_file = "predictions_async_gpt-4o-mini_drop_20250312_223527.xlsx"

# 예측 파일 로드
df = pd.read_excel(predictions_file)

# platinum_target 컬럼이 문자열 형태라면, 리스트(또는 array)로 변환
def parse_target(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            print(f"Error parsing target: {x} - {e}")
            return x
    return x

df['platinum_target'] = df['platinum_target'].apply(parse_target)

# 평가할 태스크 목록
tasks_list = ['platinum_prompt', 'platinum_prompt_no_cot', 'RE2', 'sum', 'table', 'graph', 'bullet_point', 
              'sRE2', 'RE2_no_cot', 'sum_no_cot', 'table_no_cot', 
              'graph_no_cot', 'bullet_point_no_cot', 'sRE2_no_cot']

# -------------------------------
# 최종 정답 추출 함수 (비동기)
# -------------------------------
async def extract_final_answer(session, response_text, retries=3, delay=1, model="gpt-4o", max_tokens=50, temperature=0):
    """
    chain-of-thought가 포함된 응답(response_text)에서 최종 정답만 추출합니다.
    프롬프트는 "Extract the final answer from the following response: ..."와 같이 구성됩니다.
    """
    url = "https://api.openai.com/v1/chat/completions"
    # API 키는 환경 변수 또는 코드 내에 직접 지정
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
    
    # 평가: 각 태스크별로 최종 정답이 platinum_target 배열에 포함되는지 확인 (case-insensitive 비교)
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
    
    eval_df = pd.DataFrame(list(evaluation.items()), columns=["Task", "Accuracy"])
    
    # 평가 결과 파일 이름 구성 (모델, 데이터셋, 타임스탬프 반영)
    model_name = "gpt-4o-mini"
    dataset_name = "gsm8k"  # 예측 파일 이름에서 추출한 데이터셋 이름
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_file = f"evaluation_metrics_extracted_{model_name}_{dataset_name}_{timestamp}.xlsx"
    eval_df.to_excel(evaluation_file, index=False)
    
    print("Evaluation completed.")
    print(eval_df)

if __name__ == "__main__":
    asyncio.run(main())
