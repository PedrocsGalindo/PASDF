import random
import sys
import trimesh
import numpy as np
import open3d as o3d
import yaml
import os
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))

sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]

dataset_path = os.path.join(current_dir, "../data/Thingi10K")
config_files_path = os.path.join(current_dir, "../config_files")

point_cloud_save_path = os.path.join(current_dir, "../dataset/Thingi10K/dataset")

def stl_to_pointcloud(stl_path: str, n_points: int = 200000):
    mesh = trimesh.load(stl_path, force='mesh')

    # Amostra pontos uniformemente na superfície
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)

    # Normais por face (boa aproximação pra defeitos por "empurrar pra dentro/fora")
    normals = mesh.face_normals[face_idx]

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

# amassado/calombo localizado
def gaussian_dent(pcd: o3d.geometry.PointCloud,
                  center: np.ndarray,
                  radius: float,
                  depth: float,
                  inward: bool = True):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    # distância ao centro
    d = np.linalg.norm(pts - center, axis=1)

    # máscara suave (gaussiana) -> 1 no centro, cai até ~0 no raio
    sigma = radius / 3.0
    w = np.exp(-(d**2) / (2 * sigma**2))
    w[d > radius] = 0.0

    # deslocamento ao longo da normal
    sign = -1.0 if inward else 1.0
    displacement = (sign * depth) * w[:, None] * nrm

    pts_def = pts + displacement

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_def)
    out.normals = o3d.utility.Vector3dVector(nrm)
    return out

def remove_region(pcd: o3d.geometry.PointCloud, center: np.ndarray, radius: float):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    d = np.linalg.norm(pts - center, axis=1)
    keep = d > radius

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    out.normals = o3d.utility.Vector3dVector(nrm[keep])
    return out

def random_center_from_pcd(pcd):
    pts = np.asarray(pcd.points)
    idx = np.random.randint(0, len(pts))
    return pts[idx]

def create_anomalies(pcd : o3d.geometry.PointCloud,
                     gaussian_dent_dict : dict,
                     remove_region_dict : dict
                     ) -> o3d.geometry.PointCloud:
    
        anomaly_types = ['dent', 'bulge', 'hole']
        selected_anomaly = random.choice(anomaly_types)
        center = random_center_from_pcd(pcd)

        if selected_anomaly == 'dent':
            pcd_ano = gaussian_dent(pcd, center=center, radius=gaussian_dent_dict['radius'], depth=gaussian_dent_dict['depth'], inward=True)
        elif selected_anomaly == 'bulge':
            pcd_ano = gaussian_dent(pcd, center=center, radius=gaussian_dent_dict['radius'], depth=gaussian_dent_dict['depth'], inward=False)
        elif selected_anomaly == 'hole':
            pcd_ano = remove_region(pcd, center=center, radius=remove_region_dict['radius'])
        
        return pcd_ano
    
"""
Falta:
    - salvar os point clouds com anomalias
    - salvar point clouds originais (sem anomalias) (para o teste)

"""
def main(cfg):
    print(f"Getting object paths from {dataset_path}..." \
          f"\nSaving point clouds to {point_cloud_save_path}\n")
    
    os.makedirs(point_cloud_save_path, exist_ok=True)
    obj_paths = sorted(glob(os.path.join(dataset_path, '*.stl')))

    for obj_idx, obj_path in enumerate(obj_paths):
        pcd = stl_to_pointcloud(obj_path, n_points=150000)
        obj_name = os.path.basename(obj_path).split('.')[0]
        os.makedirs(os.path.join(point_cloud_save_path, obj_name), exist_ok=True)
    
        for i in range(cfg['num_anomalies_per_model']):
            pcd_ano = create_anomalies(pcd,
                            gaussian_dent_dict=cfg['gaussian_dent'],
                            remove_region_dict=cfg['remove_region'])
            
            save_path = os.path.join(point_cloud_save_path, f"{obj_name}/{obj_name}_{i}.ply")
            o3d.io.write_point_cloud(save_path, pcd_ano)
    

if __name__ == '__main__':
    # You have to change the file name according to the data
    cfg_path = os.path.join(config_files_path, 'creat_anomalies_Thingi10K.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
    