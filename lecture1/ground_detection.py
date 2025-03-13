# """
# 지면 검출 알고리즘 예제

# 이 스크립트는 KITTI 데이터셋의 LiDAR 포인트 클라우드에서 지면을 검출하는
# RANSAC 기반 방법과 격자 기반 적응형 방법을 구현합니다.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
# import os
# import sys
# import time
# from pathlib import Path


# def load_kitti_lidar_data(file_path):
#     """
#     KITTI 데이터셋의 LiDAR .bin 파일을 로드합니다.
    
#     Args:
#         file_path: LiDAR 데이터 파일 경로
        
#     Returns:
#         points: numpy 배열, 형식은 Nx4 (x, y, z, intensity)
#     """
#     # 바이너리 파일 열기
#     with open(file_path, 'rb') as f:
#         # 파일 크기 확인
#         file_size = os.path.getsize(file_path)
#         # 각 포인트는 4개의 float32 값 (x, y, z, intensity), 1개의 float32는 4바이트
#         num_points = file_size // (4 * 4)  # 포인트 개수 = 파일크기 / (4개 값 * 4바이트)
        
#         # 구조체 형식으로 읽기
#         data = np.fromfile(f, dtype=np.float32).reshape(num_points, 4)
    
#     return data


# def filter_by_distance(points, max_dist=50.0):
#     """거리 기준으로 포인트 필터링"""
#     distances = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
#     return points[distances < max_dist]


# def fit_plane_to_points(points):
#     """
#     주어진 점들에 가장 잘 맞는 평면을 계산합니다.
    
#     Args:
#         points: 형태가 (N, 3)인 점 좌표 배열
    
#     Returns:
#         Tuple[np.ndarray, float]: (평면 법선 벡터 [a, b, c], 평면 상수 d)
#     """
#     # 점들의 중심 계산
#     centroid = np.mean(points, axis=0)
    
#     # 중심 기준으로 공분산 행렬 계산
#     centered_points = points - centroid
#     cov = np.dot(centered_points.T, centered_points) / len(points)
    
#     # 공분산 행렬의 고유벡터, 고유값 계산
#     eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
#     # 가장 작은 고유값에 해당하는 고유벡터가 평면의 법선 벡터
#     normal = eigenvectors[:, 0]
    
#     # 법선 벡터 정규화
#     normal = normal / np.linalg.norm(normal)
    
#     # ax + by + cz + d = 0에서 d 계산
#     d = -np.dot(normal, centroid)
    
#     return normal, d


# def point_to_plane_distance(points, normal, d):
#     """
#     각 점과 평면 사이의 거리를 계산합니다.
    
#     Args:
#         points: 형태가 (N, 3)인 점 좌표 배열
#         normal: 평면의 법선 벡터 [a, b, c]
#         d: 평면 방정식의 상수항
    
#     Returns:
#         np.ndarray: 각 점과 평면 사이의 거리
#     """
#     # 계산: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
#     return np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)


# def ransac_ground_detection(points, max_distance=0.1, ransac_n=3, num_iterations=100):
#     """
#     RANSAC 알고리즘을 사용하여 지면 평면을 검출합니다.
    
#     Args:
#         points: 형태가 (N, 3)인 점 좌표 배열
#         max_distance: 평면 모델에서 inlier로 간주할 최대 거리
#         ransac_n: 각 RANSAC 반복에서 사용할 무작위 샘플 수
#         num_iterations: RANSAC 반복 횟수
    
#     Returns:
#         Tuple[np.ndarray, np.ndarray]: (평면 모델 파라미터 [a, b, c, d], inlier 인덱스)
#     """
#     best_model = None
#     best_inliers = None
#     best_num_inliers = 0
    
#     for _ in range(num_iterations):
#         # 무작위 샘플링
#         indices = np.random.choice(len(points), ransac_n, replace=False)
#         sampled_points = points[indices]
        
#         # 세 점으로부터 평면 추정
#         if len(sampled_points) == 3:
#             normal, d = fit_plane_to_points(sampled_points)
            
#             # 모든 점과 평면 사이의 거리 계산
#             distances = point_to_plane_distance(points, normal, d)
            
#             # 임계값 내의 inlier 찾기
#             inliers = np.where(distances < max_distance)[0]
#             num_inliers = len(inliers)
            
#             # 더 많은 inlier를 가진 모델 업데이트
#             if num_inliers > best_num_inliers:
#                 best_num_inliers = num_inliers
#                 best_model = np.append(normal, d)
#                 best_inliers = inliers
    
#     # 최적 모델이 없으면, 기본 수평 평면 반환
#     if best_model is None:
#         best_model = np.array([0, 0, 1, 0])  # z = 0 평면
#         best_inliers = np.array([])
    
#     # 모든 inlier를 사용하여 모델 재피팅
#     if len(best_inliers) > 3:
#         normal, d = fit_plane_to_points(points[best_inliers])
#         best_model = np.append(normal, d)
    
#     return best_model, best_inliers


# def adaptive_ground_detection(points, grid_size=1.0, cell_size=10.0, height_threshold=0.2, slope_threshold=0.3):
#     """
#     격자 기반 적응형 지면 검출 방법.
#     영역을 격자로 나누어 각 셀마다 지면을 독립적으로 검출한 후 통합합니다.
#     비평탄 지형이나 언덕에서도 효과적입니다.
    
#     Args:
#         points: 형태가 (N, 3)인 점 좌표 배열
#         grid_size: 격자 셀의 크기(m)
#         cell_size: 검색 영역의 크기(m)
#         height_threshold: 지면으로 분류할 수 있는 최대 높이 차이
#         slope_threshold: 허용 가능한 최대 경사도
    
#     Returns:
#         Tuple[dict, np.ndarray]: (지면 평면 모델 딕셔너리, 지면 포인트 인덱스)
#     """
#     # 포인트 클라우드의 x, y 범위 계산
#     x_min, y_min = np.min(points[:, :2], axis=0)
#     x_max, y_max = np.max(points[:, :2], axis=0)
    
#     # 격자 인덱스 계산
#     grid_indices_x = np.floor((points[:, 0] - x_min) / grid_size).astype(int)
#     grid_indices_y = np.floor((points[:, 1] - y_min) / grid_size).astype(int)
    
#     # 격자 칸 수 계산
#     num_cells_x = int(np.ceil((x_max - x_min) / grid_size))
#     num_cells_y = int(np.ceil((y_max - y_min) / grid_size))
    
#     # 모든 격자 셀에 대해 처리
#     all_ground_indices = []
#     grid_plane_models = {}
    
#     for i in range(num_cells_x):
#         for j in range(num_cells_y):
#             # 현재 셀의 인덱스
#             cell_indices = np.where((grid_indices_x == i) & (grid_indices_y == j))[0]
            
#             if len(cell_indices) < 10:  # 포인트가 너무 적은 셀은 건너뜀
#                 continue
            
#             # 셀 중심점 계산
#             cell_center_x = x_min + (i + 0.5) * grid_size
#             cell_center_y = y_min + (j + 0.5) * grid_size
            
#             # 중심점 주변 영역의 포인트 선택 (더 넓은 영역)
#             nearby_indices = np.where(
#                 (np.abs(points[:, 0] - cell_center_x) < cell_size/2) &
#                 (np.abs(points[:, 1] - cell_center_y) < cell_size/2)
#             )[0]
            
#             if len(nearby_indices) < 50:  # 주변 포인트가 너무 적으면 건너뜀
#                 continue
            
#             # 이 영역에 대해 지면 검출 (RANSAC 사용)
#             plane_model, ground_idx = ransac_ground_detection(
#                 points[nearby_indices], 
#                 max_distance=height_threshold
#             )
            
#             # 경사도 확인 (너무 수직인 평면은 지면이 아님)
#             normal = plane_model[:3]
#             if np.abs(np.dot(normal, [0, 0, 1])) < slope_threshold:
#                 continue  # 경사가 너무 가파름
            
#             # 현재 셀에 속한 지면 포인트 인덱스
#             cell_ground_indices = nearby_indices[ground_idx]
#             cell_ground_indices = np.intersect1d(cell_ground_indices, cell_indices)
            
#             all_ground_indices.append(cell_ground_indices)
#             grid_plane_models[(i, j)] = plane_model
    
#     # 모든 셀의 지면 포인트 통합
#     if all_ground_indices:
#         ground_indices = np.concatenate(all_ground_indices)
#         ground_indices = np.unique(ground_indices)
#     else:
#         ground_indices = np.array([], dtype=int)
    
#     return grid_plane_models, ground_indices


# def visualize_ground_segmentation(points, ground_indices, title="Ground Segmentation"):
#     """지면 분할 결과 시각화"""
#     # 지면/비지면 포인트 분리
#     ground_mask = np.zeros(len(points), dtype=bool)
#     ground_mask[ground_indices] = True
    
#     # 그림 설정
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 비지면 포인트 (빨간색)
#     ax.scatter(points[~ground_mask, 0], points[~ground_mask, 1], points[~ground_mask, 2], 
#               c='r', s=0.5, alpha=0.5, label='Non-Ground Points')
    
#     # 지면 포인트 (초록색)
#     ax.scatter(points[ground_mask, 0], points[ground_mask, 1], points[ground_mask, 2], 
#               c='g', s=0.5, alpha=0.5, label='Ground Points')
    
#     # 축 라벨
#     ax.set_xlabel('X (forward)')
#     ax.set_ylabel('Y (left)')
#     ax.set_zlabel('Z (up)')
#     ax.set_title(title)
    
#     # 시점 조정
#     ax.view_init(elev=15, azim=-60)
    
#     # 범례 표시
#     ax.legend()
    
#     plt.tight_layout()
#     plt.show()


# def visualize_plane_model(points, plane_model, xlim=(-20, 20), ylim=(-20, 20), grid_size=1.0):
#     """추정된 평면 모델을 그리드로 시각화"""
#     # 그림 설정
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 포인트 시각화 (서브샘플링)
#     sample_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
#     ax.scatter(points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2], 
#               c='gray', s=0.5, alpha=0.3)
    
#     # 평면 모델 파라미터
#     a, b, c, d = plane_model
    
#     # 그리드 생성
#     x_grid = np.arange(xlim[0], xlim[1], grid_size)
#     y_grid = np.arange(ylim[0], ylim[1], grid_size)
#     x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
#     # 평면 방정식에서 z 좌표 계산: ax + by + cz + d = 0 -> z = -(ax + by + d) / c
#     if abs(c) > 1e-10:  # c가 너무 작으면 수직에 가까운 평면
#         z_mesh = -(a * x_mesh + b * y_mesh + d) / c
        
#         # 평면 시각화
#         ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.3, color='green')
#     else:
#         print("평면이 Z축에 거의 수직이므로 시각화를 건너뜁니다.")
    
#     # 축 라벨
#     ax.set_xlabel('X (forward)')
#     ax.set_ylabel('Y (left)')
#     ax.set_zlabel('Z (up)')
#     ax.set_title(f'Plane Model: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0')
    
#     # 축 범위 설정
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
    
#     # 시점 조정
#     ax.view_init(elev=25, azim=-60)
    
#     plt.tight_layout()
#     plt.show()


# def visualize_grid_plane_models(points, grid_plane_models, grid_size=5.0, xlim=(-20, 30), ylim=(-15, 15)):
#     """여러 격자 셀의 평면 모델을 시각화"""
#     # 그림 설정
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 포인트 시각화 (서브샘플링)
#     sample_idx = np.random.choice(len(points), min(3000, len(points)), replace=False)
#     ax.scatter(points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2], 
#               c='gray', s=0.5, alpha=0.2)
    
#     # 각 격자 셀에 대한 평면 시각화
#     for (i, j), plane_model in grid_plane_models.items():
#         # 평면 모델 파라미터
#         a, b, c, d = plane_model
        
#         # 격자 셀의 범위 계산
#         x_min = xlim[0] + i * grid_size
#         x_max = x_min + grid_size
#         y_min = ylim[0] + j * grid_size
#         y_max = y_min + grid_size
        
#         # 격자 내 그리드 생성
#         x_grid = np.linspace(x_min, x_max, 5)
#         y_grid = np.linspace(y_min, y_max, 5)
#         x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        
#         if abs(c) > 1e-10:
#             z_mesh = -(a * x_mesh + b * y_mesh + d) / c
            
#             # 각 셀마다 다른 색상으로 시각화
#             color = plt.cm.jet(np.random.rand())
#             ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.4, color=color)
    
#     # 축 라벨
#     ax.set_xlabel('X (forward)')
#     ax.set_ylabel('Y (left)')
#     ax.set_zlabel('Z (up)')
#     ax.set_title('Grid-based Adaptive Ground Plane Models')
    
#     # 축 범위 설정
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
    
#     # 시점 조정
#     ax.view_init(elev=30, azim=-60)
    
#     plt.tight_layout()
#     plt.show()


# def main():
#     # 데이터 경로 설정
#     # data_dir = "../data"
#     data_dir = "/workspaces/autonomous-driving-lidar-demo/data"
#     # lidar_file = os.path.join(data_dir, "sequences/00/velodyne/0000000000.bin")
#     lidar_file = os.path.join(data_dir, "velodyne_points/data/0000000000.bin")
    
#     # 파일이 존재하는지 확인
#     if not os.path.exists(lidar_file):
#         print(f"파일을 찾을 수 없습니다: {lidar_file}")
#         print("data/README.md 파일을 참조하여 KITTI 데이터셋을 다운로드하세요.")
#         return
    
#     # LiDAR 데이터 로드
#     print("LiDAR 데이터 로드 중...")
#     points = load_kitti_lidar_data(lidar_file)
#     print(f"로드된 포인트 수: {len(points)}")
    
#     # 거리 필터링 적용
#     filtered_points = filter_by_distance(points)
#     print(f"필터링 후 포인트 수: {len(filtered_points)}")
    
#     # RANSAC 지면 검출
#     print("\nRANSAC 지면 검출 실행 중...")
#     start_time = time.time()
#     plane_model, ground_indices = ransac_ground_detection(
#         filtered_points[:, :3],  # 좌표만 사용 (강도 제외)
#         max_distance=0.15,       # 지면으로 간주할 최대 거리 (미터)
#         ransac_n=3,              # 평면 추정에 사용할 포인트 수
#         num_iterations=100       # RANSAC 반복 횟수
#     )
#     end_time = time.time()
    
#     print(f"RANSAC 지면 검출 소요 시간: {end_time - start_time:.3f}초")
#     print(f"검출된 평면 모델 파라미터 [a, b, c, d]: {plane_model}")
#     print(f"지면으로 분류된 포인트 수: {len(ground_indices)}")
#     print(f"지면 포인트 비율: {len(ground_indices) / len(filtered_points) * 100:.2f}%")
    
#     # 지면과 비지면 포인트 분리
#     ground_points = filtered_points[ground_indices]
#     non_ground_indices = np.setdiff1d(np.arange(len(filtered_points)), ground_indices)
#     non_ground_points = filtered_points[non_ground_indices]
    
#     # 지면 검출 결과 시각화
#     print("\nRANSAC 지면 검출 결과 시각화...")
#     visualize_ground_segmentation(filtered_points[:, :3], ground_indices, 
#                                 title="RANSAC Ground Detection Results")
    
#     # 평면 모델 시각화
#     visualize_plane_model(filtered_points[:, :3], plane_model, xlim=(-20, 30), ylim=(-15, 15))
    
#     # 지면 제거 전/후 BEV 비교
#     plt.figure(figsize=(16, 8))
#     plt.subplot(1, 2, 1)
#     plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=0.3, c='blue', alpha=0.5)
#     plt.xlabel('X (forward)')
#     plt.ylabel('Y (left)')
#     plt.title('Before Ground Removal')
#     plt.axis('equal')
#     plt.grid(True)
    
#     plt.subplot(1, 2, 2)
#     plt.scatter(non_ground_points[:, 0], non_ground_points[:, 1], s=0.3, c='red', alpha=0.5)
#     plt.xlabel('X (forward)')
#     plt.ylabel('Y (left)')
#     plt.title('After Ground Removal')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
#     # 격자 기반 적응형 지면 검출
#     print("\n격자 기반 적응형 지면 검출 실행 중...")
#     start_time = time.time()
#     grid_plane_models, adaptive_ground_indices = adaptive_ground_detection(
#         filtered_points[:, :3],  # 좌표만 사용
#         grid_size=5.0,           # 격자 셀 크기 (미터)
#         cell_size=10.0,          # 계산에 사용할 영역 크기
#         height_threshold=0.15,   # 지면으로 간주할 최대 높이 차이
#         slope_threshold=0.3      # 허용 가능한 최대 경사도
#     )
#     end_time = time.time()
    
#     print(f"격자 기반 적응형 지면 검출 소요 시간: {end_time - start_time:.3f}초")
#     print(f"생성된 격자 셀 수: {len(grid_plane_models)}")
#     print(f"지면으로 분류된 포인트 수: {len(adaptive_ground_indices)}")
#     print(f"지면 포인트 비율: {len(adaptive_ground_indices) / len(filtered_points) * 100:.2f}%")
    
#     # 격자 기반 방법으로 지면과 비지면 포인트 분리
#     adaptive_ground_points = filtered_points[adaptive_ground_indices]
#     adaptive_non_ground_indices = np.setdiff1d(np.arange(len(filtered_points)), adaptive_ground_indices)
#     adaptive_non_ground_points = filtered_points[adaptive_non_ground_indices]
    
#     # 격자 기반 지면 검출 결과 시각화
#     print("\n격자 기반 지면 검출 결과 시각화...")
#     visualize_ground_segmentation(filtered_points[:, :3], adaptive_ground_indices, 
#                                 title="Grid-based Adaptive Ground Detection Results")
    
#     # 격자별 평면 모델 시각화
#     visualize_grid_plane_models(filtered_points[:, :3], grid_plane_models, 
#                                grid_size=5.0, xlim=(-20, 30), ylim=(-15, 15))
    
#     # 두 방법 비교: 공통/차이 지면 포인트 분석
#     common_ground = np.intersect1d(ground_indices, adaptive_ground_indices)
#     only_ransac = np.setdiff1d(ground_indices, adaptive_ground_indices)
#     only_adaptive = np.setdiff1d(adaptive_ground_indices, ground_indices)
    
#     print(f"\n두 방법 비교:")
#     print(f"공통 지면 포인트 수: {len(common_ground)}")
#     print(f"RANSAC만의 지면 포인트 수: {len(only_ransac)}")
#     print(f"적응형만의 지면 포인트 수: {len(only_adaptive)}")


# if __name__ == "__main__":
#     main()

