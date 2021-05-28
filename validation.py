import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import  RunKmeans
from .datasets import MNISTDataset, BDGPDataset, CCVDataset, HandWriteDataset
from .models import Net4Mnist, Net4BDGP, Net4HW


def extract_features(data_loader, model, device):
    model.eval()
    commonZ = []
    with torch.no_grad():
        for data in data_loader:
            Xs, y = [d.to(device) for d in data[:-1]], data[-1].to(device)
            common = model.test_commonZ(Xs)
            commonZ.extend(common.detach().cpu().numpy().tolist())

    commonZ = np.array(commonZ)
    return commonZ


def validate(data_loader, model, labels, n_clusters, device):
    commonZ = extract_features(data_loader, model, device)
    acc, nmi, pur, ari = RunKmeans(commonZ, labels, K=n_clusters, cv=5)
    return acc, nmi, pur, ari


def validate_HW():
    pass

def validate_BDGP():
    net_params = [1750, 79, 100]
    bdgp_dataset = BDGPDataset('Data/BDGP/', need_target=True)

def validate_MNIST():
    pass
