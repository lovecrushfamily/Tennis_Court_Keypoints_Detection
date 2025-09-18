import argparse
import os
import yaml
from src.train import train
from src.infer import infer
from src.utils import set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','infer','eval'], required=True)
    p.add_argument('--config', type=str, default='configs/tennis_resnet50.yaml')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--pred', type=str, default=None)
    p.add_argument('--gt', type=str, default=None)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # override device
    if args.device:
        cfg['training']['device'] = args.device

    # set seed
    seed = cfg.get('training', {}).get('seed', 42)
    set_seed(seed)

    # ensure dirs
    ensure_dir(cfg.get('logging',{}).get('tb_logdir','logs'))
    ensure_dir(cfg.get('logging',{}).get('ckpt_dir','checkpoints'))
    ensure_dir(cfg.get('outputs','outputs'))

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'infer':
        if not args.checkpoint:
            raise ValueError('Please provide --checkpoint for inference')
        infer(cfg, checkpoint=args.checkpoint)
    elif args.mode == 'eval':
        # Use script evaluate.py for evaluation
        from scripts.evaluate import evaluate_pred
        pred = args.pred or os.path.join(cfg.get('outputs','outputs'),'predicted.json')
        gt = args.gt or cfg['dataset'].get('test_json')
        evaluate_pred(gt, pred, cfg)