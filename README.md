# SmolVLM의 VQA 성능을 올려보자!
## 프로젝트 세팅
### 1. conda 가상환경 생성
```
conda create -n smolvlm python=3.10
```
```
conda activate smolvlm
```
### 2. torch 설치
CUDA 사용이 불가능한 device :
```
pip install torch
```
CUDA 사용이 가능한 device :
```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
### 3. requirements 설치
```
pip install -r requirements.txt
```
### 4. 프로젝트 실행
```
python inference_docvqa.py
```
```
python evaluate_docvqa.py
```
## 프로젝트 파일 안내
### config.yaml (modify)
LLM 모델과 데이터셋, device를 설정하는 파일

`device: "cuda"` : 필요 시 "cpu"나 "mps"로 변경  
`sample_size: 50` : 필요 시 데이터셋 개수 변경 (0으로 설정 시 전체 데이터셋 개수)  
`sample_seed: 42` : 필요 시 데이터셋 시드 변경  

### infernce.py (modify)
데이터셋을 바탕으로 모델을 실행시키는 파일

```python
from models.smolvlm import SmolVLM
# from models.model_name import ModelName
```
```python
model_classes = {
    "SmolVLM": SmolVLM,
    # "ModelName": ModelName,
}
```
필요 시 모델 추가, 모델 추가 시 import 필수

### evaluate.py (modify)
inference에서 실행한 모델의 기록을 바탕으로 모델 성능을 평가하는 파일
```python
results_file = "results/SmolVLM_DocumentVQA_validation_seed42.json"
```
평가할 파일 경로로 수정 후 평가 코드 실행행

### models/smolvlm.py
비교 대상이 될 base model 파일

### models/model_name.py (modify)
본인의 아이디어를 구현할 파일