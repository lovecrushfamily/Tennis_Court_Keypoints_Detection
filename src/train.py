import os
import time
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from src.utils import save_json
from src.data import build_dataloaders
from src.model import build_model
from src.losses import MaskedL1Loss
from src.metrics import rmse, pck


def train(cfg):
    device = torch.device(cfg['training'].get('device','cuda:0') if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = build_dataloaders(cfg, mode='train')
    model = build_model(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training'].get('lr',1e-4), weight_decay=cfg['training'].get('weight_decay',1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training'].get('epochs',40))
    scaler = GradScaler(enabled=cfg['training'].get('mixed_precision',True))
    criterion = MaskedL1Loss()

    best_rmse = 1e9
    ckpt_dir = cfg['logging'].get('ckpt_dir','checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg['training'].get('epochs',40)+1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} train')
        for batch in pbar:
            imgs = batch['image'].to(device).float()
            kps = batch['keypoints'].to(device).float()
            # Flatten: (B,2K)
            if kps.numel()==0:
                continue
            B = imgs.shape[0]
            kps = kps.view(B, -1)

            optimizer.zero_grad()
            with autocast(enabled=cfg['training'].get('mixed_precision',True)):
                preds = model(imgs)
                loss = criterion(preds, kps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # validation
        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device).float()
                kps = batch['keypoints']
                if kps.numel()==0:
                    continue
                B = imgs.shape[0]
                kps = kps.view(B,-1).numpy()
                preds = model(imgs).cpu().numpy()
                all_preds.append(preds)
                all_gts.append(kps)
        if len(all_preds)==0:
            val_rmse = None
        else:
            import numpy as np
            all_preds = np.vstack(all_preds)
            all_gts = np.vstack(all_gts)
            val_rmse = rmse(all_preds, all_gts)
            val_pck = pck(all_preds, all_gts, threshold=cfg['metrics'].get('pck_thresh',0.02))
            print(f'Epoch {epoch} val RMSE: {val_rmse:.4f} PCK: {val_pck:.4f}')

            # save best
            if val_rmse is not None and val_rmse < best_rmse:
                best_rmse = val_rmse
                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'cfg': cfg,
                    'best_rmse': best_rmse
                }
                torch.save(ckpt, os.path.join(ckpt_dir, 'best_ckpt.pth'))
                print('Saved best checkpoint')

    # final save
    torch.save({'model_state': model.state_dict(), 'cfg': cfg}, os.path.join(ckpt_dir, 'final.pth'))
    print('Training complete')