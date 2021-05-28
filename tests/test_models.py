import unittest

import torch
from torch import nn

from models import Net4Mnist, Net4BDGP, Net4HW


class TestModels(unittest.TestCase):

    def setUp(self):
        """Prepare some paramerters.
        """
        # Common parameters.
        self.batch_size = 16
        self.device = 'cpu'

    def test_Net4BDGP(self):
        net_params = [1750, 79, 200]
        model = Net4BDGP(*net_params)
        self.assertIsInstance(model, nn.Module)
        test_view1 = torch.randn(self.batch_size, 1750)
        test_view2 = torch.randn(self.batch_size, 79)
        test_common = model.test_commonZ([test_view1, test_view2])
        self.assertEqual(test_common.shape, (self.batch_size, 200))


    def test_Net4Mnist(self):
        net_params = [784, 784, 200]
        model = Net4Mnist(*net_params)
        self.assertIsInstance(model, nn.Module)
        test_view1 = torch.randn(self.batch_size, 784)
        test_view2 = torch.randn(self.batch_size, 784)
        test_common = model.test_commonZ([test_view1, test_view2])
        self.assertEqual(test_common.shape, (self.batch_size, 200))


    def test_Net4HW(self):
        net_params = [240, 76, 200]
        model = Net4HW(*net_params)
        self.assertIsInstance(model, nn.Module)
        test_view1 = torch.randn(self.batch_size, 240)
        test_view2 = torch.randn(self.batch_size, 76)
        test_common = model.test_commonZ([test_view1, test_view2])
        self.assertEqual(test_common.shape, (self.batch_size, 200))
