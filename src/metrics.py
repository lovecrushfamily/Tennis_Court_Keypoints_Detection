import numpy as np


def rmse(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    return np.sqrt(np.mean((pred-gt)**2))


def pck(pred, gt, threshold):
    # pred, gt: (N, 2K)
    pred = np.array(pred)
    gt = np.array(gt)
    N, D = pred.shape
    K = D//2
    distances = np.linalg.norm(pred.reshape(N,K,2)-gt.reshape(N,K,2), axis=2)  # (N,K)
    correct = (distances <= threshold).astype(float)
    return correct.mean()