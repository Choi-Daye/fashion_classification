## 성별 및 스타일 분류 모델

### 📌 주제
  - **성별 및 스타일 분류 모델 개발**

<br>

### 📌 데이터 셋
- **AI-Hub**
    - 연도별 패션 선호도 파악 및 추천 데이터

<br>

### 📌 실험 과정

#### 1. 정답 데이터 추출

- **사진 데이터의 제목에서 정보 추출**
    - **성별**
    - **스타일**
    - **→ 총 31가지 class로 분류**

#### 2. 탐지 성능 향상을 위한 Object Detection (YOLOv10)

- **배경 및 불필요한 인물 제거**
    - **YOLOv10을 이용해 person 객체 탐지**
    - 사진에 여러 명의 사람이 있을 경우, **가장 큰 bounding box의 영역만 남김**
        - **→ bounding box 안의 영역만 학습 데이터로 사용**

#### 3. 모델 학습

#### 이미지 전처리 (`transform`)

- **Resize:** 224x224
- **Data Augmentation**
    - 랜덤 수평 뒤집기
    - 밝기(0.2)
    - 대비(0.2)
    - 채도(0.2)
    - 랜덤 회전

#### 데이터 로드 (`DataLoader`)

- **Batch size:** 128
- **Shuffle:** True

#### 모델 구조 (`ResNet18`)

- **Weights:** None (가중치 초기화, 처음부터 학습)
- **Dropout:** 과적합 방지
- **Fully Connected Layer:** 31개 클래스 분류
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Scheduler:** StepLR
- **Epoch:** 100+100+50+100 Epoch

<br>

### 📌 결과

- **Epoch:** 350
- **Test Loss:** 1.67
- **Test Accuracy:** 64.29%

<br>

### 📌 Serving

- **Flask**
    - 웹 서버 구축
- **PyTorch**
    - **ResNet18의 마지막 레이어를 31개 클래스로 변경**
    - 저장된 모델 가중치 사용


### 📌 기술 스택

#### AI Framework
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/YOLOv10-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black">
#### Web Framework
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white">

