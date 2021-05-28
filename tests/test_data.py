import os
import unittest

from datasets import MNISTDataset, BDGPDataset, HandWriteDataset


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.BDGP_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Data', 'BDGP')
        self.MNIST_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Data', 'MNIST')
        self.HW_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Data', 'HW')

    def test_load_data(self):
        bdgp_dataset = BDGPDataset(self.BDGP_ROOT, need_target=True)
        mnist_dataset = MNISTDataset(self.MNIST_ROOT, need_target=True)
        hw_dataset = HandWriteDataset(self.HW_ROOT, need_target=True)
        self.assertIsInstance(bdgp_dataset, BDGPDataset)
        self.assertIsInstance(mnist_dataset, MNISTDataset)
        self.assertIsInstance(hw_dataset, HandWriteDataset)
        self.assertEqual(len(bdgp_dataset), 2500)
        self.assertEqual(len(mnist_dataset), 60000)
        self.assertEqual(len(hw_dataset), 2000)