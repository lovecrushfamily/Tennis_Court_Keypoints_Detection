from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, ShiftScaleRotate,
    RandomBrightnessContrast, Blur, GaussNoise, Resize
)
from albumentations.pytorch import ToTensorV2


def get_train_transforms(input_size):
    h,w = input_size
    return Compose([
        Resize(h,w),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
        RandomBrightnessContrast(p=0.5),
        Blur(p=0.2),
    ], keypoint_params={'format':'xy', 'remove_invisible':False})


def get_valid_transforms(input_size):
    h,w = input_size
    return Compose([
        Resize(h,w),
    ], keypoint_params={'format':'xy', 'remove_invisible':False})