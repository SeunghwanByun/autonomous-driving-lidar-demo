# 2차시: LiDAR와 카메라 센서 융합 및 객체 검출

이 폴더는 자율주행 LiDAR 인지 기술 강의의 두 번째 차시 자료를 포함하고 있습니다. 본 강의에서는 LiDAR와 카메라 센서 간의 캘리브레이션, 센서 융합 기법, 그리고 객체 검출 알고리즘에 대해 학습합니다.

## 학습 목표

- LiDAR와 카메라 간의 캘리브레이션 원리 및 방법 이해
- 서로 다른 센서 모달리티 간의 융합 기법 학습
- 포인트 클라우드 기반 객체 검출 알고리즘 구현
- 센서 융합을 통한 객체 검출 성능 향상 방법 이해

## 강의 자료

### 01_lidar_camera_calibration.ipynb
- 센서 캘리브레이션의 수학적 기초
- 외부 파라미터(Extrinsic Parameter)와 내부 파라미터(Intrinsic Parameter)
- KITTI 데이터셋의 캘리브레이션 파일 이해
- LiDAR-카메라 캘리브레이션 구현
- 포인트 클라우드의 카메라 이미지 투영
- 캘리브레이션 결과 검증 및 시각화

### 02_sensor_fusion.ipynb
- 센서 융합의 기본 개념과 필요성
- 초기/중기/후기 융합 방식 비교
- 데이터 레벨 융합과 특징 레벨 융합
- LiDAR와 카메라 데이터 동기화
- RGB-D 데이터 생성 및 활용
- 멀티모달 특징 추출 및 융합

### 03_object_detection.ipynb
- 객체 검출 알고리즘 개요
- 포인트 클라우드 기반 클러스터링
- 3D 바운딩 박스 생성 및 처리
- 객체 분류를 위한 기하학적 특징 추출
- 센서 융합 기반 객체 검출 구현
- 객체 검출 성능 평가

## 실습 준비

1. KITTI 데이터셋이 `../../data/kitti` 경로에 다운로드되어 있는지 확인하세요.
2. 필요한 패키지가 모두 설치되었는지 확인하세요:
```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import torch  # 딥러닝 기반 방법을 위해 필요
```

## 데이터셋 준비

본 강의에서는 KITTI Vision Benchmark Suite를 사용합니다. 데이터셋은 다음 파일들을 포함해야 합니다:
- LiDAR 포인트 클라우드 (`velodyne` 폴더)
- 카메라 이미지 (`image_2` 폴더)
- 캘리브레이션 파일 (`calib` 폴더)
- 객체 레이블(선택 사항) (`label_2` 폴더)

데이터셋 다운로드 방법은 `../../data/README.md`를 참조하세요.

## 과제

1. 제공된 KITTI 데이터를 활용하여 LiDAR와 카메라 간의 캘리브레이션을 수행하고, 포인트 클라우드를 이미지에 투영해보세요.
2. 멀티모달 데이터 융합 파이프라인을 구현하고, 각 센서의 장단점을 분석해보세요.
3. 센서 융합 기반 객체 검출 알고리즘을 구현하고, 성능을 평가해보세요.

## 참고 자료

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- Geiger, A., et al. (2013). Vision meets robotics: The KITTI dataset
- Qi, C. R., et al. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation
- Chen, X., et al. (2017). Multi-view 3D object detection network for autonomous driving
- Ku, J., et al. (2018). Joint 3D proposal generation and object detection from view aggregation

## 다음 차시 예고

다음 차시에서는 객체 추적 및 융합 인지 시스템에 대해 학습합니다. 다중 객체 추적(MOT), 칼만 필터 기반 상태 추정, 그리고 이들을 통합한 인지 시스템을 구현하는 방법을 다룰 예정입니다.