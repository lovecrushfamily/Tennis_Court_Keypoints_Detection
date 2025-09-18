# Tennis Keypoint Detection

This project provides a minimal PyTorch codebase to train a ResNet50-based keypoint regressor for tennis court keypoints.

Run training:

```bash
python run.py --mode train --config configs/tennis_resnet50.yaml
```

Run inference:

```bash
python run.py --mode infer --config configs/tennis_resnet50.yaml --checkpoint checkpoints/best_ckpt.pth
```

Evaluate:

```bash
python run.py --mode eval --config configs/tennis_resnet50.yaml --pred outputs/predicted.json --gt data/annotations/test.json
```

Notes:
- Annotations accepted in COCO-like format or as a list of dicts with `file_name` and `keypoints` fields.
- Keypoints format: either COCO `[x,y,v,...]` or flattened `[x,y,x,y,...]`.
```
