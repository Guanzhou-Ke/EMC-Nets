import os
import pickle as cPickle
import gzip

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import scipy.io


class MNISTDataset(Dataset):
    """
    Mnist-edge dataset.
    Refer by:
    Chao Shang, Aaron Palmer, Jiangwen Sun, Ko-Shin Chen, Jin Lu, and Jinbo Bi. VIGAN: missing view imputation with generative adversarial networks. CoRR, abs/1708.06724, 2017.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 need_target=False):
        self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.filename_test_domain_1 = "mnist_test_original.pickle"
        self.filename_test_domain_2 = "mnist_test_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.need_target = need_target
        self.train = train  # training set or test set
        self.download()
        self.create_two_domains()
        # now load the picked numpy arrays
        # if self.train:
        filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
        filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
        filename_test_domain_1 = os.path.join(self.root, self.filename_test_domain_1)
        filename_test_domain_2 = os.path.join(self.root, self.filename_test_domain_2)
        data_a, labels_a = cPickle.load(gzip.open(filename_train_domain_1, 'rb'))
        data_b, labels_b = cPickle.load(gzip.open(filename_train_domain_2, 'rb'))
        testdata_a, testlabels_a = cPickle.load(gzip.open(filename_test_domain_1, 'rb'))
        testdata_b, testlabels_b = cPickle.load(gzip.open(filename_test_domain_2, 'rb'))
        self.data_a = torch.Tensor(np.r_[data_a, testdata_a])
        self.labels = torch.LongTensor(np.r_[labels_a, testlabels_a])
        self.data_b = torch.Tensor(np.r_[data_b, testdata_b])

    def __getitem__(self, index):

        img_a, img_b = self.data_a[index], self.data_b[index]
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        if self.need_target:
            return img_a, img_b, self.labels[index]
        else:
            return img_a, img_b

    def __len__(self):
        return len(self.data_a)

    def download(self):
        filename = os.path.join(self.root, self.filename)
        if os.path.isfile(filename):
            return
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Download %s to %s" % (self.url, filename))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def create_two_domains(self):

        def save_domains(input_data, input_labels, domain_1_filename, domain_2_filename, domain_1_filename_test,
                         domain_2_filename_test):
            n_samples = input_data.shape[0]
            test_samples = int(n_samples / 10)
            arr = np.arange(n_samples)
            np.random.shuffle(arr)
            data_a = np.zeros((n_samples - test_samples, 1, 28, 28))
            label_a = np.zeros(n_samples - test_samples, dtype=np.int32)
            data_b = np.zeros((n_samples - test_samples, 1, 28, 28))
            label_b = np.zeros(n_samples - test_samples, dtype=np.int32)
            test_data_a = np.zeros((test_samples, 1, 28, 28))
            test_label_a = np.zeros(test_samples, dtype=np.int32)
            test_data_b = np.zeros((test_samples, 1, 28, 28))
            test_label_b = np.zeros(test_samples, dtype=np.int32)

            for i in range(0, n_samples - test_samples):
                img = input_data[arr[i], :].reshape(28, 28)
                label = input_labels[arr[i]]
                dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
                edge = dilation - img
                data_a[i, 0, :, :] = img
                data_b[i, 0, :, :] = edge
                label_a[i] = label
                label_b[i] = label

            for i in range(n_samples - test_samples, n_samples):
                img = input_data[arr[i], :].reshape(28, 28)
                label = input_labels[arr[i]]
                dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
                edge = dilation - img
                test_data_a[i - (n_samples - test_samples), 0, :, :] = img
                test_data_b[i - (n_samples - test_samples), 0, :, :] = edge
                test_label_a[i - (n_samples - test_samples)] = label
                test_label_b[i - (n_samples - test_samples)] = label

            with gzip.open(domain_1_filename, 'wb') as handle:
                cPickle.dump((data_a, label_a), handle)
            with gzip.open(domain_2_filename, 'wb') as handle:
                cPickle.dump((data_b, label_b), handle)
            with gzip.open(domain_1_filename_test, 'wb') as handle:
                cPickle.dump((test_data_a, test_label_a), handle)
            with gzip.open(domain_2_filename_test, 'wb') as handle:
                cPickle.dump((test_data_b, test_label_b), handle)

        filename = os.path.join(self.root, self.filename)
        filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
        filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
        filename_test_domain_1 = os.path.join(self.root, self.filename_test_domain_1)
        filename_test_domain_2 = os.path.join(self.root, self.filename_test_domain_2)
        if os.path.isfile(filename_train_domain_1) and os.path.isfile(filename_train_domain_2) \
                and os.path.isfile(filename_test_domain_1) and os.path.isfile(filename_test_domain_2):
            return None
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        f.close()
        # images = train_set[0]
        # labels = train_set[1]

        images = np.concatenate((train_set[0], valid_set[0]), axis=0)
        labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
        print("Compute edge images")
        print("Save origin to %s and edge to %s" % (filename_train_domain_1, filename_train_domain_2))
        save_domains(images, labels, filename_train_domain_1, filename_train_domain_2, filename_test_domain_1,
                     filename_test_domain_2)
        print("[DONE]")


class BDGPDataset(Dataset):
    """
    BDGP dataset
    Refer by:
    Xiao Cai, Hua Wang, Heng Huang, and Chris Ding. Joint stage recognition and anatomical annotation of drosophila gene expression patterns. Bioinformatics, 28(12):i16– i24, 2012.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 need_target=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.need_target = need_target
        self.paried_num = int(2500 * 1)
        data_0 = scipy.io.loadmat(os.path.join(self.root, 'paired_a2500all.mat'))
        data_dict = dict(data_0)
        self.data_a = torch.Tensor(np.array(data_dict['xpaired']))
        data_2 = scipy.io.loadmat(os.path.join(self.root, 'paired_b2500all.mat'))
        data_dict = dict(data_2)
        self.data_b = torch.Tensor(np.array(data_dict['ypaired']))
        labels = scipy.io.loadmat(os.path.join(self.root, 'label.mat'))
        labels = dict(labels)
        labels = np.array(labels['label'])
        self.labels = torch.LongTensor(labels).reshape(-1, )


    def __getitem__(self, index):
        img_a, img_b = self.data_a[index], self.data_b[index]
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        if self.need_target:
            return img_a, img_b, self.labels[index]
        else:
            return img_a, img_b

    def __len__(self):
        return self.paried_num


class HandWriteDataset(Dataset):
    """
    Hand writing dataset
    Refer by:
    Arthur Asuncion and David Newman. Uci machine learning repository, 2007, 2007.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 need_target=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.need_target = need_target

        ###### Only use mfeat-pix and mfeat-fou. You can add more dataset in here if you need.
        mfeat_pix = 'mfeat-pix'  # (2000, 240)
        mfeat_fou = 'mfeat-fou'  # (2000, 76)
        #####

        if os.path.isfile(os.path.join(self.root, f"{mfeat_pix}.npy")) and os.path.isfile(
                os.path.join(self.root, f"{mfeat_fou}.npy")) \
                and os.path.isfile(os.path.join(self.root, "labels.npy")):
            print("Load saved data...")
            self.mfeat_pix_data = torch.Tensor(np.load(os.path.join(self.root, f"{mfeat_pix}.npy")))
            self.mfeat_fou_data = torch.Tensor(np.load(os.path.join(self.root, f"{mfeat_fou}.npy")))
            self.labels = torch.LongTensor(np.load(os.path.join(self.root, "labels.npy")))
        else:
            print("Create data...")
            mfeat_pix_data, labels = self.process_raw_data(os.path.join(self.root, mfeat_pix))
            mfeat_fou_data, _ = self.process_raw_data(os.path.join(self.root, mfeat_fou))
            data = np.c_[mfeat_pix_data, mfeat_fou_data, labels]
            np.random.shuffle(data)
            mfeat_pix_data, mfeat_fou_data, labels = data[:, :240] / 255, data[:, 240:-1], data[:, -1]
            # Saved data.
            np.save(os.path.join(self.root, "labels.npy"), labels)
            np.save(os.path.join(self.root, f"{mfeat_pix}.npy"), mfeat_pix_data)
            np.save(os.path.join(self.root, f"{mfeat_fou}.npy"), mfeat_fou_data)
            self.mfeat_pix_data = torch.Tensor(mfeat_pix_data)
            self.mfeat_fou_data = torch.Tensor(mfeat_fou_data)
            self.labels = torch.LongTensor(labels)

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_a, data_b = self.mfeat_pix_data[index], self.mfeat_fou_data[index]
        if self.transform is not None:
            data_a = self.transform(data_a)
            data_b = self.transform(data_b)
        if self.need_target:
            return data_a, data_b, self.labels[index]
        else:
            return data_a, data_b,

    def process_raw_data(self, path):
        with open(path, 'r') as f:
            mfeat_data = []
            target = 0
            mfeat_labels = []
            for _ in range(1, 2001):
                raw_line = f.readline()
                raw_data = raw_line.split(' ')
                data = []
                for d in raw_data:
                    if d != '':
                        data.append(float(d))
                mfeat_data.append(data)
                mfeat_labels.append(target)
                if _ % 200 == 0:
                    target += 1
        mfeat_data = np.array(mfeat_data).astype(np.float16)
        mfeat_labels = np.array(mfeat_labels).astype(np.uint8)
        return mfeat_data, mfeat_labels


class CCVDataset(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 need_target=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.need_target = need_target

        train_label_path = 'trainLabel.txt'
        test_label_path = 'testLabel.txt'

        # n x 4000
        MFCC_train_path = 'MFCC-trainFeature.txt'
        MFCC_test_path = 'MFCC-testFeature.txt'

        # n x 5000
        STIP_trainFeature_path = 'STIP-trainFeature.txt'
        STIP_testFeature_path = 'STIP-testFeature.txt'

        # n x 5000
        SIFT_trainFeature_path = 'SIFT-trainFeature.txt'
        SIFT_testFeature_path = 'SIFT-testFeature.txt'

        n_train = 4659
        n_test = 4658

        if os.path.isfile(os.path.join(self.root, "MFCC.npy")) and \
                os.path.isfile(os.path.join(self.root, "STIP.npy")) and \
                os.path.isfile(os.path.join(self.root, "SIFT.npy")) and \
                os.path.isfile(os.path.join(self.root, "labels.npy")):
            print("Load saved data...")
            self.mfcc_data = torch.Tensor(np.load(os.path.join(self.root, "MFCC.npy")))
            self.stip_data = torch.Tensor(np.load(os.path.join(self.root, "STIP.npy")))
            self.sift_data = torch.Tensor(np.load(os.path.join(self.root, "SIFT.npy")))
            self.labels = torch.LongTensor(np.load(os.path.join(self.root, "labels.npy")))
            print("Done.")
        else:
            print("Create data...")
            # preprocess gt.
            train_labels = self.process_raw_data(os.path.join(self.root, train_label_path), n_train)
            test_labels = self.process_raw_data(os.path.join(self.root, test_label_path), n_test)
            train_ava = np.argwhere(train_labels[train_labels.sum(axis=1) == 1])
            train_ava_idx, train_ava_labels = train_ava[:, 0], train_ava[:, 1]
            test_ava = np.argwhere(test_labels[test_labels.sum(axis=1) == 1])
            test_ava_idx, test_ava_labels = test_ava[:, 0], test_ava[:, 1]
            labels = np.r_[train_ava_labels, test_ava_labels]


            # preprocess MFCC.
            train_MFCC = self.process_raw_data(os.path.join(self.root, MFCC_train_path), n_train)
            test_MFCC = self.process_raw_data(os.path.join(self.root, MFCC_test_path), n_test)
            train_MFCC, test_MFCC = train_MFCC[train_ava_idx], test_MFCC[test_ava_idx]
            mfcc_data = np.r_[train_MFCC, test_MFCC]
            mfcc_data = self.normalization(mfcc_data)


            # preprocess STIP
            train_STIP = self.process_raw_data(os.path.join(self.root, STIP_trainFeature_path), n_train)
            test_STIP = self.process_raw_data(os.path.join(self.root, STIP_testFeature_path), n_test)
            train_STIP, test_STIP = train_STIP[train_ava_idx], test_STIP[test_ava_idx]
            stip_data = np.r_[train_STIP, test_STIP]
            stip_data = self.normalization(stip_data)

            # preprocess SIFT
            train_SIFT = self.process_raw_data(os.path.join(self.root, SIFT_trainFeature_path), n_train)
            test_SIFT = self.process_raw_data(os.path.join(self.root, SIFT_testFeature_path), n_test)
            train_SIFT, test_SIFT = train_SIFT[train_ava_idx], test_SIFT[test_ava_idx]
            sift_data = np.r_[train_SIFT, test_SIFT]
            sift_data = self.normalization(sift_data)

            data = np.c_[mfcc_data, stip_data, sift_data, labels]
            np.random.shuffle(data)
            mfcc_data, stip_data, sift_data, labels = data[:, 0:4000], data[:, 4000:9000], data[:, 9000:-1], data[:, -1]
            self.labels = torch.LongTensor(labels)
            self.mfcc_data = torch.Tensor(mfcc_data)
            self.stip_data = torch.Tensor(stip_data)
            self.sift_data = torch.Tensor(sift_data)

            np.save(os.path.join(self.root, "SIFT.npy"), sift_data)
            np.save(os.path.join(self.root, "STIP.npy"), stip_data)
            np.save(os.path.join(self.root, "MFCC.npy"), mfcc_data)
            np.save(os.path.join(self.root, "labels.npy"), labels)
            print("Done.")



    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        mfcc, stip, sift = self.mfcc_data[index], self.stip_data[index], self.sift_data[index]
        if self.transform is not None:
            mfcc, stip, sift = self.transform(mfcc), self.transform(stip), self.transform(sift)
        if self.need_target:
            return mfcc, stip, sift, self.labels[index]
        else:
            return mfcc, stip, sift


    def process_raw_data(self, path, nrow):
        with open(path, 'r') as f:
            total_data = []
            for _ in range(nrow):
                raw_line = f.readline()
                if raw_line is None:
                    break
                raw_data = raw_line.split(' ')
                data = []
                for d in raw_data:
                    if d != '' and d != '\n':
                        data.append(float(d))
                total_data.append(data)
        total_data = np.array(total_data).astype(np.float16)
        return total_data

    def normalization(self, data):
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data


if __name__ == '__main__':
    ccv_path = '../../datasets/CCVdatabase/'
    ccv_dataset = CCVDataset(ccv_path, need_target=True)
    print(ccv_dataset[0])



