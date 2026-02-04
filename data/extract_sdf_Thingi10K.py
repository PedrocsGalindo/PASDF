import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))

sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]

from utils import utils_mesh
import point_cloud_utils as pcu
from glob import glob
from datetime import datetime
import yaml
import pybullet as pb
import trimesh
import gc

"""
For each object, sample points and store their distance to the nearest triangle.
"""

dataset_path = os.path.join(current_dir, "Thingi10K")
results_path = os.path.join(current_dir, "../results")
config_files_path = os.path.join(current_dir, "../config_files")

def combine_sample_latent(samples, latent_class):
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
    return np.hstack((latent_class_full, samples))

def load_stl_as_vf(stl_path):
    # load STL as mesh
    mesh_tm = trimesh.load(stl_path, force='mesh')

    # garant that is triangles
    if hasattr(mesh_tm, "triangles") is False and not isinstance(mesh_tm, trimesh.Trimesh):
        mesh_tm = utils_mesh._as_mesh(mesh_tm)

    # if came as Scene
    mesh_tm = utils_mesh._as_mesh(mesh_tm)

    verts = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces = np.asarray(mesh_tm.faces, dtype=np.int64)

    return mesh_tm, verts, faces

def main(cfg):
    # se sua pasta tiver subpastas, use **/*.stl
    obj_paths = sorted(glob(os.path.join(dataset_path, "**", "*.stl"), recursive=True))

    samples_dict = {}
    idx_str2int_dict = {}
    idx_int2str_dict = {}

    for obj_idx, obj_path in enumerate(obj_paths):
        
        obj_idx_str = os.sep.join(obj_path.split(os.sep)[-2:-1])  # pode ajustar
        idx_str2int_dict[obj_idx_str] = obj_idx
        idx_int2str_dict[obj_idx] = obj_idx_str
        samples_dict[obj_idx] = {}

        try:
            mesh_original, verts, faces = load_stl_as_vf(obj_path)

            # STL frequentily is not watertight → try fix
            if not mesh_original.is_watertight:
                verts_wt, faces_wt = pcu.make_mesh_watertight(
                    mesh_original.vertices, mesh_original.faces, 50000
                )
                mesh_original = trimesh.Trimesh(vertices=verts_wt, faces=faces_wt)
                verts = np.asarray(mesh_original.vertices, dtype=np.float64)
                faces = np.asarray(mesh_original.faces, dtype=np.int64)

        except Exception as e:
            print("Erro lendo STL:", obj_path, e)
            continue

        # (opcional) se você quiser aplicar a mesma rotação do ShapeNetAD:
        # mesh_original = utils_mesh.shapenet_rotate(mesh_original)
        # verts = np.asarray(mesh_original.vertices, dtype=np.float64)
        # faces = np.asarray(mesh_original.faces, dtype=np.int64)

        p_vol = np.random.rand(cfg['num_samples_in_volume'], 3) * 2 - 1

        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=v_min, high=v_max, size=(cfg['num_samples_in_bbox'], 3))

        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        p_total = np.vstack((p_vol, p_bbox, p_surf))

        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)

        samples_dict[obj_idx]['sdf'] = sdf
        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(
            p_total, np.array([obj_idx], dtype=np.int32)
        )

        del verts, faces, mesh_original, p_total, sdf
        gc.collect()

    os.makedirs(os.path.join(results_path, "Thingi10K"), exist_ok=True)
    np.save(os.path.join(results_path, "Thingi10K", f'samples_dict_{cfg["dataset"]}.npy'), samples_dict)
    np.save(os.path.join(results_path, "Thingi10K", "idx_str2int_dict.npy"), idx_str2int_dict)
    np.save(os.path.join(results_path, "Thingi10K", "idx_int2str_dict.npy"), idx_int2str_dict)

if __name__ == '__main__':
    cfg_path = os.path.join(config_files_path, 'extract_sdf_Thingi10K.yaml')  # crie esse yaml
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)