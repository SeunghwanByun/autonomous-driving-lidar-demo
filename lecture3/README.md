# 3차시: LiDAR 인지와 상태 추정 통합 전략

이 폴더는 자율주행 LiDAR 인지 기술 강의의 세 번째 차시 자료를 포함하고 있습니다. 본 강의에서는 다중 객체 추적, 칼만 필터 기반 상태 추정, 그리고 통합 인지 시스템 구현까지 자율주행 인지의 고급 기술을 다룹니다.

## 학습 목표

- 다중 객체 추적(MOT) 알고리즘의 원리와 구현 방법을 이해한다
- 칼만 필터 기반 상태 추정 기법을 익히고 적용할 수 있다
- 자율주행을 위한 통합 인지 시스템을 설계하고 구현할 수 있다
- 인지 시스템의 성능 평가 및 최적화 방법을 알아본다

## 강의 자료

### 01_multi_object_tracking.ipynb
- 다중 객체 추적의 개념과 과제
- 데이터 연관(Data Association) 알고리즘 구현
- 헝가리안 알고리즘 및 IoU 기반 매칭
- ID 관리 및 트랙 생성/소멸 전략
- 3D 객체 추적의 특징 및 도전 과제

### 02_kalman_filter.ipynb
- 베이지안 필터링의 기초 이론
- 칼만 필터의 원리 및 구현
- 운동 모델 설계 (등속 모델, 등가속 모델)
- 확장 칼만 필터 및 무향 칼만 필터 개요
- 파티클 필터와의 비교 및 적용 사례

### 03_integrated_perception.ipynb
- 자율주행을 위한 통합 인지 파이프라인 설계
- 센서 융합, 객체 검출, 추적의 통합 구현
- 시간 동기화 및 지연 문제 해결 전략
- 실시간 처리를 위한 최적화 기법
- 인지 시스템 성능 평가 방법론

## 실습 준비

1. KITTI 데이터셋이 `../../data/kitti` 경로에 다운로드되어 있는지 확인하세요.
2. 필요한 패키지가 모두 설치되었는지 확인하세요:
```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import filterpy
```

## 과제

1. 제공된 객체 검출 결과를 활용하여 다중 객체 추적 시스템을 구현하세요.
2. 칼만 필터를 사용하여 객체의 위치와 속도를 예측하고 추정 성능을 평가하세요.
3. 센서 융합, 객체 검출, 추적을 결합한 통합 인지 시스템을 구현하고 결과를 시각화하세요.

## 참고 자료

- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004). Estimation with applications to tracking and navigation.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking.
- Welch, G., & Bishop, G. (1995). An introduction to the Kalman filter.
- Weng, X., Wang, J., Held, D., & Kitani, K. (2020). AB3DMOT: A Baseline for 3D Multi-Object Tracking.

## 강의 요약
이번 차시에서는 객체 검출 결과를 활용하여 객체를 시간에 따라 지속적으로 추적하는 기술과 불확실성이 존재하는 상황에서 객체의 상태를 정확히 추정하는 기법을 학습했습니다. 또한 이러한 개별 기술들을 자율주행 시스템에 맞게 통합하는 방법을 살펴보았습니다. 향후 실제 자율주행 시스템 개발 시 활용할 수 있는 전체 인지 파이프라인의 기초를 이해하고 구현해보았습니다.