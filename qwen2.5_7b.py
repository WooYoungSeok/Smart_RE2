import os
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# 장치 설정: CUDA 사용 가능 여부에 따라 device 결정
if torch.cuda.is_available():
    device = "cuda"
    print("Using device: cuda:0")
else:
    device = "cpu"
    print("Using device: cpu")

########################################
# (1) 모델, 토크나이저 로드
########################################
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
# pad_token_id 없을 경우 설정
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
########################################
# (3) 데이터셋 로드 및 전처리
######################################## 
if len(sys.argv) < 2:
    print("Usage: python qwen2.5_7b.py <data_file>")
    sys.exit(1)
data_file = sys.argv[1]
# python qwen2.5_7b.py Smart_RE2/sRE2_datasets/navigate_add_task.parquet

# Parquet 파일 로드 후 'cleaning_status'가 "rejected"인 행 제거
df = pd.read_parquet(data_file)
df = df[df["cleaning_status"] != "rejected"].reset_index(drop=True)
print(f"'{data_file}' 파일에서 {len(df)} 건의 데이터가 로드되었습니다.")

# 사용될 태스크(데이터 열) 리스트 (프로젝트 파일과 동일)
tasks_list = [
    'platinum_prompt', 'platinum_prompt_no_cot', 
    'RE2', 'sRE2', 'RE2_no_cot', 'sRE2_no_cot'
]

########################################
# (4) 각 태스크별 추론 실행 및 결과 저장
########################################
for task in tasks_list:
    predictions = []
    print(f"\nProcessing task: {task}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Task {task}"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{row[task]}"}
        ]
        
         # 토크나이저의 챗 템플릿 기능을 사용해 프롬프트 생성
        prompt = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # 프롬프트가 딕셔너리라면 각 요소를 device로 이동하고, attention_mask가 누락되었다면 생성합니다.
        if isinstance(prompt, dict):
            prompt = {key: value.to(device) for key, value in prompt.items()}
            if "attention_mask" not in prompt:
                prompt["attention_mask"] = (prompt["input_ids"] != tokenizer.pad_token_id).long().to(device)
        else:
            # 만약 프롬프트가 단일 텐서라면 device 이동 후 attention_mask 생성
            prompt = prompt.to(device)
            attention_mask = (prompt != tokenizer.pad_token_id).long()
            prompt = {"input_ids": prompt, "attention_mask": attention_mask}
        
        # 결정론적 디코딩: input_ids와 함께 attention_mask도 전달합니다.
        output_ids = model.generate(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            max_new_tokens=2054,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        predictions.append(generated_text)
    
    # 각 태스크 결과를 DataFrame의 새로운 열로 추가
    df[f"{task}_pred"] = predictions
    time.sleep(1)  # 간단한 딜레이


# 결과 파일명 생성: 입력 파일 이름(첫번째 '_' 이전)과 타임스탬프 포함
dataset_name = os.path.basename(data_file).split("_")[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"predictions_qwen_{dataset_name}_{timestamp}.xlsx"

# 예측 결과를 엑셀 파일로 저장
df.to_excel(output_file, index=False)
print(f"\nPredictions saved to '{output_file}'.")