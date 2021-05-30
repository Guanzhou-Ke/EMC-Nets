import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import RunKmeans
from datasets import MNISTDataset, BDGPDataset, CCVDataset, HandWriteDataset
from models import Net4Mnist, Net4BDGP, Net4HW


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


def validate_BDGP():
    print(f"{'-'*30} Begin validation BDGP {'-'*30}")
    net_params = [1750, 79, 200]
    bdgp_dataset = BDGPDataset('./Data/BDGP/', need_target=True)
    net = Net4BDGP(*net_params)
    net.load_state_dict(torch.load('./checkpoints/model_weight4BDGP.pth'))
    bdgp_loader = DataLoader(bdgp_dataset, batch_size=128)
    commonZ = extract_features(bdgp_loader, net, 'cpu')
    RunKmeans(commonZ, bdgp_loader.dataset.labels.numpy(), K=5, cv=5)
    print(f"{'-' * 30} End validation BDGP {'-' * 30}")

def validate_MNIST():
    print(f"{'-' * 30} Begin validation MNIST {'-' * 30}")
    net_params = [784, 784, 200]
    mnist_dataset = MNISTDataset('./Data/MNIST/', need_target=True)
    net = Net4Mnist(*net_params)
    net.load_state_dict(torch.load('./checkpoints/model_weight4MNIST.pth'))
    mnist_loader = DataLoader(mnist_dataset, batch_size=128)
    commonZ = extract_features(mnist_loader, net, 'cpu')
    RunKmeans(commonZ, mnist_loader.dataset.labels.numpy(), K=10, cv=5)
    print(f"{'-' * 30} End validation MNIST {'-' * 30}")


def validate_HW():
    print(f"{'-' * 30} Begin validation HW {'-' * 30}")
    net_params = [240, 76, 200]
    hw_dataset = HandWriteDataset('./Data/HW/', need_target=True)
    net = Net4HW(*net_params)
    net.load_state_dict(torch.load('./checkpoints/model_weight4HW.pth'))
    hw_loader = DataLoader(hw_dataset, batch_size=128)
    commonZ = extract_features(hw_loader, net, 'cpu')
    RunKmeans(commonZ, hw_loader.dataset.labels.numpy(), K=10, cv=5)
    print(f"{'-' * 30} End validation HW {'-' * 30}")


if __name__ == '__main__':
    validate_BDGP()
    validate_MNIST()
    validate_HW()