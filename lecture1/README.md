# 1차시: 자율주행과 LiDAR 인지 기술의 개요

이 폴더는 자율주행 LiDAR 인지 기술 강의의 첫 번째 차시 자료를 포함하고 있습니다. 본 강의에서는 LiDAR 센서의 기본 원리부터 포인트 클라우드 처리, 노면 검출, 차선 인식까지 자율주행 인지의 기초를 다룹니다.

## 학습 목표

- LiDAR 센서의 작동 원리와 특성을 이해한다
- 포인트 클라우드 데이터의 구조와 처리 방법을 익힌다
- 노면 검출 알고리즘의 원리와 구현 방법을 배운다
- 포인트 클라우드 기반 차선 인식 기법을 구현할 수 있다

## 강의 자료

### 01_lidar_basics.ipynb
- LiDAR 센서 작동 원리와 종류 (기계식, 솔리드 스테이트, FMCW 등)
- 포인트 클라우드 데이터 구조 및 형식 (PCD, PLY, BIN 등)
- KITTI 데이터셋 로딩 및 시각화
- 점군 필터링 및 전처리 기법 (다운샘플링, 노이즈 제거)
- 좌표계 변환 및 데이터 처리

### 02_ground_detection.ipynb
- 노면 검출의 중요성과 활용
- RANSAC 기반 노면 검출 알고리즘
- 높이 기반 노면 검출 방법
- 노면 제거 및 장애물 세그멘테이션
- 노면 모델링 및 파라미터 추출

### 03_lane_detection.ipynb
- LiDAR 기반 차선 인식의 원리
- 반사율(Intensity) 정보 활용 기법
- 바닥면 투영 및 누적 히스토그램 방법
- 차선 검출 알고리즘 구현
- 곡선 차선 모델링 및 파라미터 추출

## 실습 준비

1. KITTI 데이터셋이 `../../data/kitti` 경로에 다운로드되어 있는지 확인하세요.
2. 필요한 패키지가 모두 설치되었는지 확인하세요:
```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
```

## 과제

1. 제공된 KITTI 데이터셋의 포인트 클라우드를 로드하고 시각화하세요.
2. RANSAC 알고리즘을 활용하여 노면을 검출하고, 노면 제거 후 장애물만 시각화하세요.
3. 노면의 반사율(Intensity) 정보를 활용하여 차선을 검출하는 알고리즘을 구현하세요.

## 참고 자료

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite.
- Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019). SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences.
- Himmelsbach, M., Hundelshausen, F. V., & Wuensche, H. J. (2010). Fast segmentation of 3d point clouds for ground vehicle navigation.

## 다음 차시 예고

다음 차시에서는 LiDAR와 카메라 센서 융합 및 객체 검출에 대해 학습합니다. LiDAR-카메라 캘리브레이션, 센서 융합 기법, 그리고 객체 검출 알고리즘 등을 다룰 예정입니다.