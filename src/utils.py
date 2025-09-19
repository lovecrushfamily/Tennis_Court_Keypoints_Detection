import os
import random
import json
import numpy as np
import torch
import cv2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def visualize_keypoints(image, keypoints, save_path=None, radius=3, color=(0,255,0)):
    # keypoints: [x1,y1,x2,y2,...]
    img = image.copy()
    for i in range(0, len(keypoints), 2):
        x = int(round(keypoints[i]))
        y = int(round(keypoints[i+1]))
        cv2.circle(img, (x,y), radius, color, -1)
    if save_path:
        cv2.imwrite(save_path, img)
    return img