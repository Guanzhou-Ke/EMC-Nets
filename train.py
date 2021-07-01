import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import do_clustering, seed_everything, RunKmeans, AverageMeter, normalized_mutual_info_score, accuracy, \
    save_dict
from datasets import MNISTDataset, BDGPDataset, CCVDataset, HandWriteDataset
from models import Net4Mnist, Net4BDGP, Net4HW


def create_data_loader(datasets, batch_size, num_workers, init=False, labels=None):
    if init:
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)
    if labels is not None:
        datasets.labels = labels
        datasets.need_target = True
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)
    else:
        datasets.need_target = False
        return DataLoader(datasets, batch_size=batch_size, num_workers=num_workers)


def extract_features(train_loader, model, device):
    model.eval()
    commonZ = []
    with torch.no_grad():
        for data in train_loader:
            Xs, y = [d.to(device) for d in data[:-1]], data[-1].to(device)
            common = model.test_commonZ(Xs)
            commonZ.extend(common.detach().cpu().numpy().tolist())

    commonZ = np.array(commonZ)
    return commonZ


def validate(data_loader, model, labels_holder, n_clusters, device):
    commonZ = extract_features(data_loader, model, device)
    acc, nmi, pur, ari = RunKmeans(commonZ, labels_holder['labels_gt'], K=n_clusters, cv=1)
    return acc, nmi, pur, ari


def unsupervised_clustering_step(model, train_loader, num_workers, labels_holder, n_clusters, device):
    print('[Pesudo labels]...')
    features = extract_features(train_loader, model, device)

    if 'labels' in labels_holder:
        labels_holder['labels_prev_step'] = labels_holder['labels']

    if 'score' not in labels_holder:
        labels_holder['score'] = -1

    labels = do_clustering(features, n_clusters)
    labels_holder['labels'] = labels
    nmi = 0
    # score = unsupervised_measures(features, labels)
    # print(labels.shape, labels_holder['labels_gt'].shape)

    nmi_gt = normalized_mutual_info_score(labels_holder['labels_gt'], labels)
    print('NMI t / GT = {:.4f}'.format(nmi_gt))

    if 'labels_prev_step' in labels_holder:
        nmi = normalized_mutual_info_score(labels_holder['labels_prev_step'], labels)
        print('NMI t / t-1 = {:.4f}'.format(nmi))

    train_loader = create_data_loader(train_loader.dataset, train_loader.batch_size, num_workers, labels=labels)

    return train_loader, nmi_gt, nmi


def train_unsupervised(train_loader, model, optimizer, epoch, max_steps, device, tag='unsupervised', verbose=1):
    losses = AverageMeter()

    model.train()
    if verbose == 1:
        pbar = tqdm(total=len(train_loader),
                    ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[:-1]]

        loss = model.get_loss(Xs)
        losses.update(loss.item(), Xs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )
    if verbose == 1:
        pbar.close()

    return losses.avg


def train(train_loader, model, optimizer, epoch, max_steps, device, tag='train', verbose=1):
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    if verbose == 1:
        pbar = tqdm(total=len(train_loader), ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs, target = [d.to(device) for d in data[:-1]], data[-1].to(device)

        # compute output
        outputs = model(Xs)
        loss = model.get_loss(Xs, target)

        # measure accuracy and record loss
        prec1 = accuracy(outputs, target, topk=(1,))[0]  # returns tensors!
        losses.update(loss.item(), Xs[0].size(0))
        acc.update(prec1.item(), Xs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                Acc=f"{acc.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )

    if verbose == 1:
        pbar.close()

    return acc.avg, losses.avg


def main(Net, mparams, datasets, batch_size=128,
         n_clusters=10, seed=10,
         max_steps=1000, recluster_epoch=1,
         validate_epoch=1, max_unsupervised_steps=3, max_supervised_steps=2, model_path='./', verbose=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('CudNN:', torch.backends.cudnn.version())
    print('Run on {} GPUs'.format(torch.cuda.device_count()))

    start_epoch = 0
    best_nmi = 0
    ### Data loading ###
    num_workers = 4

    print('[TRAIN]...')
    seed_everything(seed)
    model = Net(*mparams).to(device)
    cls_optimizer = model.get_cls_optimizer()
    recon_optimizer = model.get_recon_optimizer()
    ###############################################################################

    labels_holder = {}  # utility container to save labels from the previous clustering step
    train_loader = create_data_loader(datasets, batch_size, num_workers, init=True, labels=None)
    labels_holder['labels_gt'] = train_loader.dataset.labels.numpy()
    history = {}
    # Training Start

    history['best_acc'] = 0
    history['nmi_gt'] = []
    history['nmi_t_1'] = []
    history['recon_loss'] = []
    history['cls_loss'] = []
    history['cluster_result'] = []
    best_score = 0

    for epoch in range(max_steps):
        nmi_gt = None

        for u_epoch in range(max_unsupervised_steps):
            loss_avg = train_unsupervised(train_loader, model, recon_optimizer, u_epoch, max_unsupervised_steps,
                                          device, verbose=verbose)
            history['recon_loss'].append(loss_avg)

        if epoch == start_epoch or epoch % recluster_epoch == 0:
            train_loader, nmi_gt, nmi_t_1 = \
                unsupervised_clustering_step(model, train_loader, num_workers, labels_holder, n_clusters, device)
            history['nmi_gt'].append(nmi_gt)
            history['nmi_t_1'].append(nmi_t_1)

        for u_epoch in range(max_supervised_steps):
            acc_avg, loss_avg = train(train_loader, model, cls_optimizer, u_epoch, max_supervised_steps, device, verbose=verbose)
            history['cls_loss'].append(loss_avg)

        if (epoch + 1) % validate_epoch == 0:
            acc, nmi, pur, ari = validate(train_loader, model, labels_holder, n_clusters, device)
            history['cluster_result'].append((acc, nmi, pur, ari))
            if acc > best_score:
                best_score = acc
                history['best_acc'] = best_score
                torch.save(model.state_dict(), model_path)
        print(f"{'-' * 20} {seed}: best score: {best_score} {'-' * 20}")
    return history


def train_BDGP(BDGP_root, seed=0):
    dataset = BDGPDataset(BDGP_root, need_target=True)
    net_params = [1750, 79, 200]
    record = main(Net4BDGP, net_params, dataset, batch_size=16,
                  n_clusters=5, max_steps=1000, model_path='bdgp_model.pth',
                  recluster_epoch=1, seed=seed,
                  validate_epoch=1, verbose=0)
    return record

def train_HW(HW_root, seed=0):
    dataset = HandWriteDataset(HW_root, need_target=True)
    net_params = [240, 76, 100]
    record = main(Net4HW, net_params, dataset, batch_size=16,
                  n_clusters=10, max_steps=1000, model_path='hw_model.pth',
                  recluster_epoch=1, seed=seed,
                  validate_epoch=1, verbose=0)
    return record

def train_MNIST(MNIST_root, seed=0):
    dataset = MNISTDataset(MNIST_root, need_target=True)
    net_params = [784, 784, 200]
    record = main(Net4Mnist, net_params, dataset, batch_size=16,
                  n_clusters=10, max_steps=1000, model_path='mnist_model.pth',
                  recluster_epoch=1, seed=seed,
                  validate_epoch=1, verbose=0)
    return record


if __name__ == '__main__':
    #----------------- BDGP dataset -------------------------
    bdgp_record = train_BDGP('Data/BDGP/', seed=4028)
    save_dict(bdgp_record, 'bdgp_training_log.json')

    # ----------------- HW dataset -------------------------
    hw_record = train_HW('Data/HW/', seed=2883)
    save_dict(hw_record, 'hw_training_log.json')

    # ----------------- MNIST dataset -------------------------
    mnist_record = train_HW('Data/MNIST/', seed=189)
    save_dict(mnist_record, 'mnist_training_log.json')

