import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils import load_json


class KeypointsDataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None, input_size=(256,256)):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.input_size = input_size

        data = load_json(ann_file)
        # Support COCO-like or simple list
        if 'images' in data and 'annotations' in data:
            images = {im['id']: im for im in data['images']}
            anns = {}
            for a in data['annotations']:
                img_id = a['image_id']
                if img_id not in anns:
                    anns[img_id] = []
                anns[img_id].append(a)
            self.samples = []
            for img_id, img in images.items():
                file_name = img['file_name']
                width = img.get('width', None)
                height = img.get('height', None)
                kp = None
                if img_id in anns and len(anns[img_id])>0:
                    # take the first annotation
                    a = anns[img_id][0]
                    kp = a.get('keypoints', None)
                self.samples.append({'file_name':file_name,'width':width,'height':height,'keypoints':kp,'image_id':img_id})
        else:
            # assume list of dicts
            self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, s['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        h0,w0 = img.shape[:2]

        keypoints = s.get('keypoints', None)
        if keypoints is None:
            keypoints_xy = []
        else:
            # assume COCO style [x1,y1,v1,...]
            if len(keypoints) % 3 == 0:
                kp_xy = []
                for i in range(0, len(keypoints), 3):
                    kp_xy.extend([keypoints[i], keypoints[i+1]])
                keypoints_xy = kp_xy
            else:
                keypoints_xy = keypoints

        # apply transforms
        if self.transforms:
            # albumentations expects keypoints as list of (x,y)
            kps = [(keypoints_xy[i], keypoints_xy[i+1]) for i in range(0, len(keypoints_xy), 2)] if len(keypoints_xy)>0 else []
            transformed = self.transforms(image=img, keypoints=kps)
            img = transformed['image']
            kps_t = transformed.get('keypoints', [])
            keypoints_xy = []
            for (x,y) in kps_t:
                keypoints_xy.extend([x,y])
        else:
            # resize to input_size
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))

        img = img.astype('float32') / 255.0
        # normalize mean/std (ImageNet)
        mean = np.array([0.485,0.456,0.406])[None,None,:]
        std = np.array([0.229,0.224,0.225])[None,None,:]
        img = (img - mean) / std
        img = img.transpose(2,0,1).copy()

        # scale keypoints to the resized image coordinates
        if len(keypoints_xy) > 0:
            # if transforms applied, they already match resized size
            kp_arr = np.array(keypoints_xy, dtype=np.float32)
        else:
            kp_arr = np.zeros((0,), dtype=np.float32)

        sample = {
            'image': img,
            'keypoints': kp_arr,
            'meta': {
                'file_name': s['file_name'],
                'original_size': (w0,h0),
                'image_id': s.get('image_id', idx)
            }
        }
        return sample


def build_dataloaders(cfg, mode='train'):
    dataset_cfg = cfg['dataset']
    input_size = tuple(dataset_cfg.get('input_size',(256,256)))
    from src.augmentations import get_train_transforms, get_valid_transforms

    if mode == 'train':
        train_ds = KeypointsDataset(dataset_cfg['train_images'], dataset_cfg['train_json'], transforms=get_train_transforms(input_size), input_size=input_size)
        val_ds = KeypointsDataset(dataset_cfg['valid_images'], dataset_cfg['valid_json'], transforms=get_valid_transforms(input_size), input_size=input_size)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training'].get('num_workers',4), pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training'].get('num_workers',4), pin_memory=True)
        return train_loader, val_loader
    else:
        test_ds = KeypointsDataset(dataset_cfg['test_images'], dataset_cfg['test_json'], transforms=get_valid_transforms(input_size), input_size=input_size)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training'].get('num_workers',4), pin_memory=True)
        return test_loader