
---

## 📝 대회 개요

- **목표:**  
  섞여 있는 4개의 한국어 문장에 대해, 가장 자연스러운 문장 순서를 예측하는 AI 알고리즘 개발


## 💻 실행 환경

### ✅ 로컬 환경

- OS: Ubuntu 22.04  
- Python: 3.11  
- GPU: NVIDIA GeForce RTX 4060 Ti  
- CPU: AMD Ryzen 5 2600

### ✅ Colab 환경

- 런타임: Google Colab Pro  
- GPU: NVIDIA A100
- 
## 🚀 실행 방법

### 1. 데이터 다운로드

- [데이터 페이지](https://dacon.io/competitions/official/236489/data)

### 2. 환경 세팅

```bash
pip install -r requirements.txt
```

### 3. 모델별 추론 결과 생성

- 각 모델별로 아래와 같이 실행하여 csv파일 결과를 생성합니다.

```bash
python src/exp21.py
python src/exp22.py
python src/exp36.py
```

- (notebook/exp19.ipynb는 Colab A100환경에서 훈련 및 추론)

### 4. 앙상블(보팅) 추론

- 여러 모델의 예측 결과를 앙상블하여 최종 결과를 생성합니다.

```bash
python src/voting.py
```

- 최종 결과(`results/ensemble.csv`)를 제출 파일로 사용합니다.

---

## 📒 참고 사항

- 각 모델별 코드 및 실험 노트북은 `src/`, `notebook/` 폴더에 정리되어 있습니다.
- 데이터 파일: [데이터 페이지](https://dacon.io/competitions/official/236489/data)
- 자세한 솔루션 설명 및 실험 결과는 `docs/` 폴더의 PDF/PPTX 파일을 참고해주세요.

---
