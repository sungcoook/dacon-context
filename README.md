
---

## 📝 대회 개요

- **목표:**  
  섞여 있는 4개의 한국어 문장에 대해, 가장 자연스러운 문장 순서를 예측하는 AI 알고리즘 개발

- **데이터 구성:**  
  - `train.csv`: sentence_0~3(섞인 문장), answer_0~3(정답 순서)
  - `test.csv`: sentence_0~3(섞인 문장)
  - `sample_submission.csv`: 제출 양식

---

## 🚀 실행 방법

### 1. 데이터 다운로드

- [데이터 페이지](https://dacon.io/competitions/official/236489/data)에서 `train.csv`, `test.csv`, `sample_submission.csv`를 다운로드하여 프로젝트 폴더(예: `data/`)에 저장합니다.

### 2. 환경 세팅

```bash
pip install -r requirements.txt
```

### 3. 모델별 추론 결과 생성

- 각 모델별로 아래와 같이 실행하여 결과(`results/exp21.csv` 등)를 생성합니다.

```bash
python src/exp21.py
python src/exp22.py
python src/exp36.py
```

- (Colab 환경에서 실행한 경우, `notebook/` 폴더 참고)

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
