import json

import numpy as np
import torch
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, adjusted_rand_score

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)


def RunKmeans(X, y, K, cv=5):
    results = []
    for _ in range(cv):
        print(f"CV {_ + 1} Beginning...")
        keams = KMeans(n_clusters=K)
        keams.fit(X)
        y_pred = keams.labels_
        results.append(measure_cluster(y_pred, y))
    results = np.array(results).mean(axis=0)
    print(f"After ron {cv} times, final Acc.: {results[0] * 100:.2f}% NMI: {results[1] * 100:.2f}% purity: \
{results[2] * 100:.2f}% and ARI: {results[3] * 100:.2f}%")
    return results


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def measure_cluster(y_pred, y_true):
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    cm = confusion_matrix(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    row_max = cm.max(axis=1).sum()
    total = cm.sum()
    pur = row_max / total
    print(f"Acc.: {acc * 100:.2f}% NMI: {nmi * 100:.2f}% purity: {pur * 100:.2f}% and ARI: {ari * 100:.2f}%")
    return acc, nmi, pur, ari


def train_kmeans(x, num_clusters=10, num_gpus=1):
    """
    Runs k-means clustering on one or several GPUs
    """
    d = x.shape[1]
    kmeans = faiss.Clustering(d, num_clusters)
    kmeans.verbose = True
    kmeans.niter = 20

    kmeans.max_points_per_centroid = 100000

    res = [faiss.StandardGpuResources() for i in range(num_gpus)]

    flat_config = []
    for i in range(num_gpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if num_gpus == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(num_gpus)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    kmeans.train(x, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids)

    stats = kmeans.iteration_stats
    objective = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])

    print(("Final objective: %.4g" % objective[-1]))

    return centroids.reshape(num_clusters, d)


def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()


def do_clustering(features, num_clusters, num_gpus=None):
    if num_gpus is None:
        num_gpus = faiss.get_num_gpus()
    features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
    centroids = train_kmeans(features, num_clusters, num_gpus)
    labels = compute_cluster_assignment(centroids, features)
    return labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_dict(obj, path):
    try:
        with open(path, 'w') as f:
            save_dict = {}
            for key in obj.keys():
                if isinstance(obj[key], list):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], int):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], np.ndarray):
                    save_dict[key] = obj[key].tolist()
            json.dump(save_dict, f, indent=4)
            print(f'Saved dict at {path}')
    except Exception as e:
        print(e)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)