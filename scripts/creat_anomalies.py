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

dataset_pcd_save_path = os.path.join(current_dir, "../dataset/Thingi10K/dataset/pcd")

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

    d = np.linalg.norm(pts - center, axis=1)

    sigma = radius / 3.0
    w = np.exp(-(d**2) / (2 * sigma**2))
    w[d > radius] = 0.0

    sign = -1.0 if inward else 1.0
    displacement = (sign * depth) * w[:, None] * nrm
    pts_def = pts + displacement

    mask = (w > 0.0).astype(np.float32)  

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_def)
    out.normals = o3d.utility.Vector3dVector(nrm)
    return out, mask

def remove_region(pcd, center, radius, thickness=0.05):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    d = np.linalg.norm(pts - center, axis=1)
    keep = d > radius

    pts_kept = pts[keep]
    nrm_kept = nrm[keep]
    d_kept = d[keep]

    band = (d_kept <= (radius + thickness)).astype(np.float32)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_kept)
    out.normals = o3d.utility.Vector3dVector(nrm_kept)
    return out, band

def random_center_from_pcd(pcd):
    pts = np.asarray(pcd.points)
    idx = np.random.randint(0, len(pts))
    return pts[idx]

def create_anomalies(pcd, gaussian_dent_dict, remove_region_dict):
    anomaly_types = ['dent', 'bulge', 'hole']
    selected_anomaly = random.choice(anomaly_types)
    center = random_center_from_pcd(pcd)

    if selected_anomaly == 'dent':
        pcd_ano, mask = gaussian_dent(
            pcd, center=center,
            radius=gaussian_dent_dict['radius'],
            depth=gaussian_dent_dict['depth'],
            inward=True
        )

    elif selected_anomaly == 'bulge':
        pcd_ano, mask = gaussian_dent(
            pcd, center=center,
            radius=gaussian_dent_dict['radius'],
            depth=gaussian_dent_dict['depth'],
            inward=False
        )

    elif selected_anomaly == 'hole':  
        thickness = remove_region_dict.get('thickness', 0.05 * remove_region_dict['radius'])
        pcd_ano, mask = remove_region(
            pcd, center=center,
            radius=remove_region_dict['radius'],
            thickness=thickness
        )

    return pcd_ano, mask, selected_anomaly

def save_gt_txt(gt_path: str, pcd: o3d.geometry.PointCloud, mask: np.ndarray):
    pts = np.asarray(pcd.points)
    assert len(pts) == len(mask), "GT mask precisa ter o mesmo tamanho do point cloud"

    data = np.hstack([pts, mask.reshape(-1, 1)])
    np.savetxt(gt_path, data, fmt="%.6f %.6f %.6f %.1f")

def main(cfg):
    print(f"Getting object paths from {dataset_path}..." \
          f"\nSaving point clouds to {dataset_pcd_save_path}\n")
    
    os.makedirs(dataset_pcd_save_path, exist_ok=True)
    obj_paths = sorted(glob(os.path.join(dataset_path, '*.stl')))

    for obj_idx, obj_path in enumerate(obj_paths):
        pcd = stl_to_pointcloud(obj_path, n_points=150000)
        obj_name = os.path.basename(obj_path).split('.')[0]
        os.makedirs(os.path.join(dataset_pcd_save_path, f"{obj_name}/test"), exist_ok=True)
        os.makedirs(os.path.join(dataset_pcd_save_path, f"{obj_name}/GT"), exist_ok=True)
    
        for i in range(cfg['num_of_rounds']):
            if random.random() < cfg['anomaly_probability']:
                pcd_ano, mask, anomaly_type = create_anomalies(pcd,
                                gaussian_dent_dict=cfg['gaussian_dent'],
                                remove_region_dict=cfg['remove_region'])
                
                stem = f"{obj_name}_{i}_{anomaly_type}"
                pcd_path = os.path.join(dataset_pcd_save_path, f"{obj_name}/test/{stem}.pcd")
                gt_path  = os.path.join(dataset_pcd_save_path, f"{obj_name}/GT/{stem}.txt")

                o3d.io.write_point_cloud(pcd_path, pcd_ano)
                save_gt_txt(gt_path, pcd_ano, mask)
            else:
                # No need to creat a GT file for normal samples (but the name need to have "positive")
                save_path = os.path.join(dataset_pcd_save_path, f"{obj_name}/test/{obj_name}_{i}_positive.pcd")
                o3d.io.write_point_cloud(save_path, pcd)

if __name__ == '__main__':
    # You have to change the file name according to the data
    cfg_path = os.path.join(config_files_path, 'creat_anomalies_Thingi10K.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
    