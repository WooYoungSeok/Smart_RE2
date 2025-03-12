import os
import json
import tqdm
import pandas as pd
from datetime import datetime
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
# (2) 실행 시 입력으로 받는 태스크 인자 처리
########################################
if sys.argv[1] == 'emobank':
    task = "emobank"
elif sys.argv[1] == 'goemotion':
    task = "goemotion"
elif sys.argv[1] == 'semeval_task1':
    task = "semeval_task1"
elif sys.argv[1] == 'semeval_task2':
    task = "semeval_task2"
elif sys.argv[1] == 'SST':
    task = "SST"
elif sys.argv[1] == 'sst5':
    task = "sst5"
elif sys.argv[1] == 'TDT':
    task = "TDT"
elif sys.argv[1] == 'vader':
    task = "vader"
else:
    raise ValueError("올바른 태스크 이름이 아닙니다.")

########################################
# (3) 배치 크기 설정 (기본값 32)
########################################
if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])
else:
    batch_size = 32  # 기본 32개씩 호출

########################################
# (4) 파일 경로 및 데이터 로드
########################################
data_file = f'C:\\Users\\dssal\\OneDrive\\바탕 화면\\emoA_baselines\\data_AEB_2\\{task}.json'
prompts_file = 'C:\\Users\\dssal\\OneDrive\\바탕 화면\\emoA_baselines\\prompts.json'

# prompts.json 파일에서 프롬프트 템플릿 로드
with open(prompts_file, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# 데이터 로드
with open(data_file, 'r', encoding='utf-8') as f:
    emobank_data = [json.loads(line) for line in f]
    # emobank_data = emobank_data[:64]  # 테스트용 (64개만)

########################################
# (5) 배치 단위로 모델 추론 (한번에 32건씩) 함수
########################################
def generate_batch_responses(batch_data, max_new_tokens=512):
    """
    batch_data(복수개의 entry)를 한꺼번에 처리하여
    한 번의 model.generate() 호출로 batch_size만큼 답변을 얻는다.
    """
    # 5-1) system_prompt, user_prompt 각각 만들기
    system_prompts = []
    user_prompts   = []
    
    for entry in batch_data:
        sp = "You are a helpful assistant.\n" + prompts['LLMs']['system'].format(Task=entry['Task'])
        up = prompts['LLMs']['user'].format(Text=entry['Text'])

        # 'cot' 옵션이 들어있다면 체인-오브-생각(cot)을 유도하는 문구 추가
        if 'cot' in sys.argv:
            up += "\nLet's think step by step."
        
        system_prompts.append(sp)
        user_prompts.append(up)

    # 5-2) 메시지를 한꺼번에 apply_chat_template
    text_list = []
    for sp, up in zip(system_prompts, user_prompts):
        messages = [
            {"role": "system", "content": sp},
            {"role": "user",   "content": up}
        ]
        # Qwen 전용 함수로, prompt 생성
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        text_list.append(text)

    # 5-3) 여러 개 텍스트를 한 번에 토크나이즈
    model_inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # 5-4) 한 번의 generate 호출로 batch 전체 추론
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1
        )

    # 5-5) 입력 부분(input_ids) 제외하고 생성 부분만 추출 & 디코딩
    responses = []
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
        gen_ids = output_ids[len(input_ids):]
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(resp)

    # 5-6) 결과를 정리해서 반환
    results = []
    for entry, sp, up, resp in zip(batch_data, system_prompts, user_prompts, responses):
        result = {
            "system": sp,
            "user": up,
            "response": resp,
            "output": entry.get("output", "N/A")
        }
        results.append(result)

    return results

########################################
# (6) 메인 함수
########################################
def main():
    all_results = []

    # tqdm로 batch 단위 진행 상황 확인
    for i in tqdm.trange(0, len(emobank_data), batch_size, desc="Processing", unit="batch"):
        batch_data = emobank_data[i : i + batch_size]
        batch_results = generate_batch_responses(batch_data)
        all_results.extend(batch_results)

    # 샘플 결과 콘솔 출력
    # for example in all_results[:5]:
    #     print(example)

    # 결과를 CSV 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_csv = f'./cot_{task}_{model_name}_{timestamp}.csv'
    os.makedirs(os.path.dirname(output_file_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_file_csv, index=False, encoding='utf-8-sig')
    print("텍스트 생성 및 결과 저장이 완료되었습니다.")

########################################
# (7) 메인 실행부
########################################
if __name__ == "__main__":
    main()
