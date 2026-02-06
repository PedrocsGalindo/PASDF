# -*- coding: utf-8 -*-
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import csv
# from utils.io_utils import set_seed
# from utils.metrics import auroc
# from utils import infer
import argparse, os, yaml, numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import Dataset_Real3D_AD_test
from dataset import Dataset_ShapeNetAD_test
from dataset import Dataset_Thingi10K_test
from utils import set_seed
from sklearn.metrics import roc_auc_score
from infer import SDFScorer
from tqdm import tqdm
import time
from utils import register_point_clouds
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dataset(cfg: dict, cls_name: str):
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name == "Real3D_AD":
        return Dataset_Real3D_AD_test(
            dataset_dir=ds_cfg["dataset_dir"],
            cls_name=cls_name,
            num_points=ds_cfg.get("num_points", 0),
            normalize=ds_cfg.get("normalize", True),
            scale_factor=ds_cfg.get("scale_factor", 17.744022369384766),
            template_path = ds_cfg.get("template_path", None)
        )
    elif name == "ShapeNetAD":
        return Dataset_ShapeNetAD_test(
            dataset_dir=ds_cfg["dataset_dir"],
            cls_name= cls_name,
            num_points=ds_cfg.get("num_points", 0),
            normalize=ds_cfg.get("normalize", False),
            scale_factor=ds_cfg.get("scale_factor", 1.0),
            template_path = ds_cfg.get("template_path", None),
        )
    elif name == "Thingi10K":
        return Dataset_Thingi10K_test(
            dataset_dir=ds_cfg["dataset_dir"],
            cls_name= cls_name,
            num_points=ds_cfg.get("num_points", 0),
            normalize=ds_cfg.get("normalize", False),
            scale_factor=ds_cfg.get("scale_factor", 1.0),
            template_path = ds_cfg.get("template_path", None),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

def evaluate(loader, cfg, cls_name):
    model = SDFScorer(cfg, cls_name, device=DEVICE)
    model.eval()

    out_dir = os.path.join(cfg["infer"]["output_dir"], "tmp_scores", cls_name)
    os.makedirs(out_dir, exist_ok=True)

    pixel_files = []
    label_files = []
    obj_scores, obj_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating [{cls_name}]", ncols=100)):
            pts, mask, label, template_path = batch["points"], batch["mask"], batch["label"], batch["template_points"]

            pts_for_infer = pts[0]
            pts_for_infer, _, _ = register_point_clouds(
                pts_for_infer, template_path[0],
                voxel_size=cfg['infer']['voxel_size'],
                cd_threshold=cfg['infer']['cd_threshold']
            )

            pts_for_infer = torch.from_numpy(np.asarray(pts_for_infer.points)).float().unsqueeze(0).to(DEVICE)

            pixel_score, object_score = model.infer(pts_for_infer)

            # salva pixel scores/labels em disco (por batch)
            ps = np.asarray(pixel_score, dtype=np.float32)
            yl = np.asarray(mask[0].flatten(), dtype=np.uint8)

            pf = os.path.join(out_dir, f"pixel_scores_{i:06d}.npy")
            lf = os.path.join(out_dir, f"pixel_labels_{i:06d}.npy")
            np.save(pf, ps); np.save(lf, yl)

            pixel_files.append(pf)
            label_files.append(lf)

            obj_scores.append(float(object_score[0]))
            obj_labels.append(int(label[0]))

            # libera referências (ajuda em cloud)
            del pts_for_infer, ps, yl, pixel_score, object_score

    # carrega tudo no final (aqui pode escolher concatenar por partes)
    pixel_scores = np.concatenate([np.load(f) for f in pixel_files])
    pixel_labels = np.concatenate([np.load(f) for f in label_files])

    pix_auc = roc_auc_score(pixel_labels, pixel_scores)
    obj_auc = roc_auc_score(obj_labels, obj_scores)
    return pix_auc, obj_auc

def main(args):
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    set_seed(cfg["seed"])
    results = []
    for cls_name in cfg['dataset']['cls_name']:
            
        ds = build_dataset(cfg, cls_name)
        loader = DataLoader(ds, batch_size=cfg['infer']['batch_size'], shuffle=cfg['infer']['shuffle'], num_workers=cfg['infer']['num_workers'])
        try:
            pix_auc, obj_auc = evaluate(loader, cfg, cls_name)
        except Exception as e:
            # aqui você captura qualquer exceção durante a avaliação
            error_message = f"erro grave na avaliação: {type(e).__name__} - {e}"
            print(error_message, file=sys.stderr) # imprime o erro no stderr
            # você pode até salvar mais detalhes se quiser, tipo um traceback completo
            import traceback
            print("\nfull traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


        print(f"---{cls_name}-- AUROC Pixel: {pix_auc}, AUROC Object: {obj_auc}")
        results.append({
            "class": cls_name,
            "pixel_auc": pix_auc,
            "object_auc": obj_auc
        })


    output_dir = cfg['infer'].get('output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "pixel_auc", "object_auc"])
        writer.writeheader()
        writer.writerows(results)

    print("\n==== Overall Evaluation Results ====")
    print(f"{'Class':<20} {'Pixel AUROC':<15} {'Object AUROC':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['class']:<20} {r['pixel_auc']:<15.4f} {r['object_auc']:<15.4f}")
    print("-" * 50)
    print(f"✅ Results saved to: {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args)
