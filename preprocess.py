"""
This script extracts 2D and 3D keypoints from 2D detections using the 4DHuman model.
Please refer to the https://github.com/shubham-goel/4D-Humans/tree/main for installation instructions.

Author: Tianjian Jiang
Date: March 16, 2025
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils import recursive_to


class Human4D:
    """A wrapper around the 4DHuman model to extract SMPL(X) parameters from 2D detections.
    It will also try to realign the SMPL to the user-provided focal length.
    4DHuman uses SMPL (neutral) with 10 shape parameters and 69 pose parameters.
    """
    def __init__(self, device):
        download_models(CACHE_DIR_4DHUMANS)
        model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.model_cfg = model_cfg
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def infer(self, img, boxes):
        img_cv2 = img[:, :, ::-1].copy()
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        temp = defaultdict(list)
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            keys = ["pred_cam", "pred_cam_t", "focal_length", "pred_keypoints_2d", "pred_keypoints_3d"]
            for k in keys:
                temp[k].append(out[k].float().cpu().numpy())
            keys = ["box_center", "box_size", "personid", "img_size"]
            for k in keys:
                temp[k].append(batch[k].float().cpu().numpy())
            for k in out["pred_smpl_params"].keys():
                temp[k].append(out["pred_smpl_params"][k].float().cpu().numpy())
        for k in temp.keys():
            temp[k] = np.concatenate(temp[k], axis=0)
        return temp

    def postprocess(self, pred):
        NUM_PERSONS = len(pred["personid"])
        kpt_2d = np.zeros((NUM_PERSONS, 25, 2))
        kpt_3d = np.zeros((NUM_PERSONS, 25, 3))
        for person_id in pred["personid"]:
            person_id = int(person_id)
            cx, cy = pred["box_center"][person_id]
            bbox_size = pred["box_size"][person_id]

            pred_keypoints_2d = pred["pred_keypoints_2d"][person_id]
            pred_keypoints_2d = pred_keypoints_2d * bbox_size + (cx, cy)
            kpt_2d[person_id] = pred_keypoints_2d[:25]

            pred_keypoints_3d = pred["pred_keypoints_3d"][person_id]
            kpt_3d[person_id] = pred_keypoints_3d[:25]
        return kpt_2d, kpt_3d

    def __call__(self, img, boxes):
        pred = self.infer(img, boxes)
        return self.postprocess(pred)

def run_eval(model, boxes, image_dir):
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    skels_2d = np.zeros((NUM_FRAMES, NUM_PERSONS, 25, 2))
    skels_3d = np.zeros((NUM_FRAMES, NUM_PERSONS, 25, 3))
    skels_2d.fill(np.nan)
    skels_3d.fill(np.nan)
    for frame in (pbar := trange(NUM_FRAMES, desc=f"{image_dir.stem}")):
        img = Image.open(image_dir / f"{frame:05d}.jpg")
        img = np.asarray(img)
        skels_2d[frame], skels_3d[frame] = model(img, boxes[frame])
    return skels_2d, skels_3d

def main(root):
    skels_2d = {}
    skels_3d = {}
    boxes = np.load(root / "boxes.npz")
    model = Human4D("cuda")
    for cam_path in root.glob("cameras/*.npz"):
        sequence_name = cam_path.stem
        image_dir = root / "images" / sequence_name
        skel_2d, skel_3d = run_eval(model, boxes[sequence_name], image_dir)
        skels_2d[sequence_name] = skel_2d
        skels_3d[sequence_name] = skel_3d
    np.savez_compressed(root / "skel_2d.npz", **skels_2d)
    np.savez_compressed(root / "skel_3d.npz", **skels_3d)

if __name__ == "__main__":
    main(Path("data/"))
