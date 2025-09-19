import os
import torch
from src.data import build_dataloaders
from src.model import build_model
from src.utils import save_json, visualize_keypoints


def infer(cfg, checkpoint):
    device = torch.device(cfg['training'].get('device','cuda:0') if torch.cuda.is_available() else 'cpu')
    test_loader = build_dataloaders(cfg, mode='infer')
    model = build_model(cfg).to(device)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck.get('model_state', ck))
    model.eval()
    outputs = []
    out_dir = cfg.get('outputs','outputs')
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device).float()
            metas = batch['meta']
            preds = model(imgs).cpu().numpy()
            for i in range(preds.shape[0]):
                meta = metas[i]
                file_name = meta['file_name']
                orig_w, orig_h = meta['original_size']
                # preds are in resized coords (input size). Need to scale back to original image size
                input_h, input_w = cfg['dataset'].get('input_size', [256,256])
                scale_x = orig_w / input_w
                scale_y = orig_h / input_h
                kps = preds[i].tolist()
                kps_scaled = []
                for j in range(0,len(kps),2):
                    x = kps[j] * scale_x
                    y = kps[j+1] * scale_y
                    kps_scaled.extend([float(x), float(y)])
                outputs.append({'file_name': file_name, 'keypoints': kps_scaled})
                # visualization
                try:
                    import cv2
                    img_path = os.path.join(cfg['dataset']['test_images'], file_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        vis = visualize_keypoints(img, kps_scaled, save_path=os.path.join(out_dir, f'vis_{file_name}'))
                except Exception:
                    pass
    save_json(outputs, os.path.join(out_dir, 'predicted.json'))
    print('Saved predictions to', os.path.join(out_dir, 'predicted.json'))