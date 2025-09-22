from src.utils import load_json
from src.metrics import rmse, pck
import numpy as np


def evaluate_pred(gt_path, pred_path, cfg=None):
    gt = load_json(gt_path)
    pred = load_json(pred_path)
    # assume both are list of dicts with file_name and keypoints
    gt_map = {item['file_name']: item for item in gt}
    preds = []
    gts = []
    for p in pred:
        fn = p['file_name']
        if fn not in gt_map:
            continue
        kp_pred = np.array(p['keypoints'])
        kp_gt = np.array(gt_map[fn]['keypoints'])
        preds.append(kp_pred)
        gts.append(kp_gt)
    if len(preds)==0:
        print('No matching predictions')
        return
    preds = np.vstack(preds)
    gts = np.vstack(gts)
    print('RMSE:', rmse(preds, gts))
    thresh = cfg['metrics'].get('pck_thresh', 0.02) if cfg else 0.02
    # interpret threshold as fraction of image diag; approximate using width from gt if provided
    # here use absolute pixels if threshold >1
    if thresh <= 1.0:
        # need image size - fallback to 100
        threshold_px = thresh * 100
    else:
        threshold_px = thresh
    print('PCK (approx):', pck(preds, gts, threshold_px))