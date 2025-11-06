import sys
import os
import torch
import yaml
import numpy as np
import trimesh
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]

import model.model_sdf as sdf_model
from utils import embed_kwargs, Embedder, extract_mesh, get_volume_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def predict_sdf_ofo_PoseEmbedder_reconstruct(coords_batches, model):
    sdf = torch.empty((0, 1), dtype=torch.float32, device=device)
    embedder = Embedder(**embed_kwargs)
    model.eval()
    for coords in coords_batches:
        coords = embedder.embed(coords).float()
        sdf_batch = model(coords)
        sdf = torch.vstack((sdf, sdf_batch))
    return sdf


def load_training_settings(checkpoint_root, class_name=None):
    cand_paths = []
    cand_paths.append(os.path.join(checkpoint_root, "settings.yaml"))
    if class_name is not None:
        cand_paths.append(os.path.join(checkpoint_root, class_name, "settings.yaml"))
    for p in cand_paths:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
    raise FileNotFoundError(f"settings.yaml not found:\n - " + "\n - ".join(cand_paths))


def extract_and_save_mesh(sdf, grad_size_axis, save_path):
    try:
        vertices, faces = extract_mesh(grad_size_axis, sdf)
    except Exception as e:
        print(f"Mesh extraction failed: {e}")
        return False
    mesh = trimesh.Trimesh(vertices, faces)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trimesh.exchange.export.export_mesh(mesh, save_path, file_type='obj')
    return True


def reconstruct_one(class_name, checkpoint_root, resolution, mesh_save_dir):
    training_settings = load_training_settings(checkpoint_root, class_name=class_name)

    weights = os.path.join(checkpoint_root, class_name, "weights.pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Weight file not found: {weights}")

    model = sdf_model.SDFModel_ofo_PoseEmbedder(
        num_layers=training_settings['num_layers'],
        skip_connections=training_settings['skip_connections'],
        inner_dim=training_settings['inner_dim'],
        PoseEmbedder_size=60
    ).float().to(device)

    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)

    coords, grad_size_axis = get_volume_coords(resolution)
    coords = coords.to(device)

    coords_batches = torch.split(coords, 100000)

    sdf = predict_sdf_ofo_PoseEmbedder_reconstruct(coords_batches, model)

    obj_path = os.path.join(mesh_save_dir, f"mesh_{class_name}.obj")
    ok = extract_and_save_mesh(sdf, grad_size_axis, obj_path)
    if ok:
        print(f"[OK] {class_name} -> {obj_path}")
    else:
        print(f"[FAIL] {class_name}")


def main(cfg_path):
    with open(cfg_path, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg_dir = os.path.dirname(cfg_path)
    project_root = os.path.dirname(cfg_dir)
    ckpt_path_cfg = cfg["checkpoint_path"]
    checkpoint_root = ckpt_path_cfg if os.path.isabs(ckpt_path_cfg) else os.path.normpath(
        os.path.join(project_root, ckpt_path_cfg)
    )

    mesh_save_dir = cfg["mesh_save_dir"]

    obj_ids = cfg["obj_ids"]
    resolution = int(cfg.get("resolution", 256))

    print(f"Project root: {project_root}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Resolution: {resolution}")
    print(f"Number of objects: {len(obj_ids)}")

    os.makedirs(mesh_save_dir, exist_ok=True)

    for class_name in obj_ids:
        try:
            reconstruct_one(class_name, checkpoint_root, resolution, mesh_save_dir)
        except Exception as e:
            print(f"[ERROR] {class_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D meshes.")
    parser.add_argument('cfg_path', type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
