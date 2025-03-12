import os
import json
import tqdm
import pandas as pd
from datetime import datetime
import sys
from transformers import pipeline
from huggingface_hub import login  # Hugging Face 인증 모듈 추가
from transformers import logging

# 로그 수준을 ERROR로 설정하여 불필요한 메시지 숨기기
logging.set_verbosity_error()

# -----------------------------
# (1) Task 설정 및 검증
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python baseline_llama3.1_CW.py <task_name> [batch_size]")
    sys.exit(1)

task = sys.argv[1]
valid_tasks = ['emobank', 'goemotion', 'semeval_task1', 'semeval_task2', 'SST', 'sst5', 'TDT', 'vader']

if task not in valid_tasks:
    print("Invalid task provided!")
    sys.exit(1)

# 배치 크기 설정 (기본값 32)
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

HF_TOKEN = "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"

# Hugging Face 인증
try:
    login(HF_TOKEN)
    print("Hugging Face authentication successful.")
except Exception as e:
    print(f"Authentication failed: {e}")
    sys.exit(1)

# -----------------------------
# (2) 모델 및 파이프라인 설정
# -----------------------------
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto"
)

# -----------------------------
# (3) 파일 경로 및 데이터 로드
# -----------------------------
data_file = f'C:\\Users\\dssal\\OneDrive\\바탕 화면\\emoA_baselines\\data_AEB_2\\{task}.json'
prompts_file = 'C:\\Users\\dssal\\OneDrive\\바탕 화면\\emoA_baselines\\prompts.json'

# prompts.json 파일 로드
with open(prompts_file, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# 데이터 로드
with open(data_file, 'r', encoding='utf-8') as f:
    emobank_data = [json.loads(line) for line in f]

# -----------------------------
# (4) 배치 처리 함수
# -----------------------------
def process_batch(batch):
    """
    주어진 배치(batch) 데이터를 모델을 사용해 처리합니다.
    """
    results = []
    for entry in batch:
        # 프롬프트 구성
        system_prompt = "You are a helpful assistant."
        user_prompt = (
            prompts['LLMs']['system'].format(Task=entry['Task']) +
            prompts['LLMs']['user'].format(Text=entry['Text'])
        )
        
        if 'cot' in sys.argv:
            user_prompt += "\nLet's think step by step."
        else:
            system_prompt += "\nAnswer in a single word."
        
        # 입력 텍스트 구성
        input_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 모델 호출
        #print(f"Processing entry: {entry['Task']}")
        outputs = pipeline(
            input_text,
            max_new_tokens=512
        )

        generated_text = outputs[0]["generated_text"][-1]

        # 결과 저장
        results.append({
            "system": system_prompt,
            "user": user_prompt,
            "response": generated_text,
            "output": entry.get("output", "N/A")
        })

    return results

# -----------------------------
# (5) 메인 함수
# -----------------------------
def main():
    all_results = []

    # tqdm을 사용해 배치 단위로 진행 상황 표시
    for i in tqdm.trange(0, len(emobank_data), batch_size, desc="Processing", unit="batch"):
        batch = emobank_data[i:i + batch_size]
        batch_results = process_batch(batch)
        all_results.extend(batch_results)

    # 결과를 CSV 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_csv = f'./{task}_{model_id}_{timestamp}.csv'
    os.makedirs(os.path.dirname(output_file_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_file_csv, index=False, encoding='utf-8-sig')
    print("텍스트 생성 및 결과 저장이 완료되었습니다.")

# -----------------------------
# (6) 실행부
# -----------------------------
if __name__ == "__main__":
    main()
