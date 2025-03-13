# autonomous-driving-lidar-demo

# 자율주행 LiDAR 인지 기술 강의 자료

이 레포지토리는 자율주행 LiDAR 인지 기술에 대한 3회차 강의를 위한 코드와 자료를 포함합니다.

## 레포지토리 구조

```
lidar-perception-lecture/
├── README.md
├── requirements.txt
├── data/
│   └── README.md (데이터 다운로드 방법 및 구조 설명)
├── lecture1/
│   ├── README.md
│   ├── 01_lidar_basics.ipynb
│   ├── 02_ground_detection.ipynb
│   └── 03_lane_detection.ipynb
├── lecture2/
│   ├── README.md
│   ├── 01_lidar_camera_calibration.ipynb
│   ├── 02_sensor_fusion.ipynb
│   └── 03_object_detection.ipynb
├── lecture3/
│   ├── README.md
│   ├── 01_multi_object_tracking.ipynb
│   ├── 02_kalman_filter.ipynb
│   └── 03_integrated_perception.ipynb
└── utils/
    ├── __init__.py
    ├── visualization.py
    ├── calibration.py
    ├── ground_detection.py
    ├── object_detection.py
    └── tracking.py
```

## 설치 및 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/your-username/lidar-perception-lecture.git
cd lidar-perception-lecture

# 필요 패키지 설치
pip install -r requirements.txt

# KITTI 데이터셋 다운로드 (data/README.md 참조)
```

## 강의 내용

### 1차시: 자율주행과 LiDAR 인지 기술의 개요
- LiDAR 기본 원리와 데이터 구조 이해
- 포인트 클라우드 시각화 및 기본 처리
- 노면 검출 및 차선 인식 알고리즘

### 2차시: LiDAR와 카메라 센서 융합 및 객체 검출
- LiDAR-카메라 캘리브레이션 및 융합 기법
- 센서 융합 기반 객체 검출 알고리즘
- 객체 형태 추정 및 활용

### 3차시: LiDAR 인지와 상태 추정 통합 전략
- 다중 객체 추적(MOT) 알고리즘
- 칼만 필터 기반 상태 추정
- 통합 인지 시스템 구현

## 데이터셋

본 강의에서는 KITTI Vision Benchmark Suite를 사용합니다. 데이터 다운로드 및 구조에 대한 설명은 `data/README.md`를 참조하세요.