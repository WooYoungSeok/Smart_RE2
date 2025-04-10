import sys
import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from huggingface_hub import login
from datetime import datetime
import torch

# 장치 설정: CUDA 사용 가능 여부에 따라 device 결정
if torch.cuda.is_available():
    device = "cuda"
    print("Using device: cuda:0")
else:
    device = "cpu"
    print("Using device: cpu")

# python llama3.1_8b.py Smart_RE2/sRE2_adaptive_datasets/drop_m.parquet
# 커맨드라인 인자로 데이터 파일 경로를 받음
if len(sys.argv) < 2:
    print("Usage: python llama3.1_8b.py <data_file>")
    sys.exit(1)
data_file = sys.argv[1]

# Parquet 파일 로드 및 'cleaning_status'가 "rejected"인 행 제거
df = pd.read_parquet(data_file)
df = df[df["cleaning_status"] != "rejected"].reset_index(drop=True)

# 처리할 태스크 리스트 (DataFrame의 열 이름과 동일)
tasks_list = [
    'platinum_prompt', 'platinum_prompt_no_cot',
    'RE2', 'sRE2', 'RE2_no_cot', 'sRE2_no_cot'
]

# Hugging Face 인증 (토큰 필요)
HF_TOKEN = "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"
try:
    login(HF_TOKEN)
    print("Hugging Face authentication successful.")
except Exception as e:
    print(f"Authentication failed: {e}")
    sys.exit(1)

# 불필요한 로깅 메시지 제거
logging.set_verbosity_error()

# 모델 ID 설정
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 토크나이저와 모델 직접 로드 (pipeline 대신 직접 생성)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 모델 설정 업데이트: greedy decoding 설정 (temperature, top_p)
model.config.temperature = 1.0
model.config.top_p = 1.0

# 각 태스크별 예측 수행
for task in tasks_list:
    predictions = []
    print(f"\nProcessing task: {task}")
    
    # DataFrame의 각 행에 대해 예측을 진행합니다.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Task {task}"):
        # system 메시지와 user 메시지로 구성된 메시지 리스트를 생성합니다.
        # user 메시지는 해당 task에 해당하는 열의 값을 사용하며, 추가로 태스크 이름을 명시합니다.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{row[task]}"}
        ]
        
        # tokenizer.apply_chat_template을 사용하여 프롬프트 텐서를 생성합니다.
        prompt = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        
        # 모델 추론: 생성된 프롬프트에 대해 텍스트를 생성합니다.
        output_ids = model.generate(
            prompt,
            max_new_tokens=2054,
            do_sample=False,  # 결정론적 출력 (greedy decoding)
            temperature=1.0,
            top_p=1.0,
        )
        
        # 생성된 토큰을 디코딩하여 최종 텍스트로 변환합니다.
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        predictions.append(generated_text)
    
    # 각 태스크별 예측 결과를 DataFrame에 새로운 열로 추가합니다.
    df[f"{task}_pred"] = predictions
    # 태스크 간 간단한 딜레이 (원하는 경우)
    time.sleep(1)

# 결과 파일명 생성: 데이터셋명(파일명 첫 번째 '_' 이전)과 타임스탬프 포함
dataset_name = os.path.basename(data_file).split("_")[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_predictions_file = f"predictions_llama3.1_{dataset_name}_{timestamp}.xlsx"

# 예측 결과를 엑셀 파일로 저장
df.to_excel(output_predictions_file, index=False)
print(f"\nPredictions saved to '{output_predictions_file}'.")
