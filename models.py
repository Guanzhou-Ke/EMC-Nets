import math
from itertools import chain

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun

    def test_commonZ(self, Xs):
        raise NotImplementedError

    def forward(self, Xs):
        raise NotImplementedError

    def decoder(self, latent):
        raise NotImplementedError

    def get_loss(self, Xs, labels=None):
        raise NotImplementedError

    def get_cls_optimizer(self):
        raise NotImplementedError

    def get_recon_optimizer(self):
        raise NotImplementedError


class Net4BDGP(BaseNet):

    def __init__(self, input_A, input_B, nz):
        super().__init__()
        self.input_A = input_A
        self.input_B = input_B
        self.nz = nz
        self.encoder1 = nn.Sequential(
            nn.Linear(input_A, 500, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(500, 300, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz, bias=True),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_B, 300, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(300, self.nz, bias=True),
            nn.ReLU(True),
        )

        # Add Transformer.
        self.trans_enc = nn.TransformerEncoderLayer(d_model=self.nz * 2, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

        self.cls_layer = nn.Sequential(
            nn.Linear(self.nz * 2, 5),
            nn.Sigmoid()
        )

        self.layer4 = nn.Linear(self.nz * 2, 300)
        self.layer5_1 = nn.Linear(300, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(300, input_B)
        self.drop = 0.5

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init('xavier'))
        self.flatten = nn.Flatten()
        self.recon_criterion = nn.BCELoss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun

    def forward(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        x = x.mean(dim=0)
        y = self.cls_layer(x)

        return y

    def decoder(self, latent):
        x = F.dropout(F.relu(self.layer4(latent)), self.drop)

        out1 = F.relu(self.layer5_1(x))
        out1 = self.sigmoid(self.layer6_1(out1))
        out2 = self.sigmoid(self.layer6_2(x))
        return out1.view(-1, self.input_A), out2.view(-1, self.input_B)

    def get_loss(self, Xs, labels=None):
        if labels is not None:
            y = self(Xs)
            cls_loss = self.cls_criterion(y, labels)
            return cls_loss
        else:
            latent = self.test_commonZ(Xs)
            recon1, recon2 = self.decoder(latent)
            recon_loss = 0.3 * self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
                         0.7 * self.recon_criterion(recon1, Xs[0]).mean(0).sum()
            return recon_loss

    def test_commonZ(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        latent = x.mean(dim=0)
        return latent

    def get_cls_optimizer(self):
        self.cls_optimizer = torch.optim.SGD(chain(self.encoder1.parameters(), self.encoder2.parameters(),
                                                   self.extract_layers.parameters(), self.cls_layer.parameters()),
                                             lr=1e-3,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        return self.cls_optimizer

    def get_recon_optimizer(self):
        self.recon_optimizer = torch.optim.SGD(self.parameters(),
                                               lr=1e-3,
                                               momentum=0.9,
                                               weight_decay=5e-4,
                                               )
        return self.recon_optimizer


class Net4Mnist(BaseNet):

    def __init__(self, input_A, input_B):
        super().__init__()
        # input_A = 784  input_B = 784
        # self.layer1_1 = nn.Linear(input_A, 400, bias=False)
        self.encoder1 = nn.Sequential(
            nn.Linear(784, 400, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(400, 500, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(500, 100, bias=True),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(784, 400, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(400, 500, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(500, 100, bias=True),
            nn.ReLU(True),
        )


        # Add Transformer.
        self.trans_enc = nn.TransformerEncoderLayer(d_model=200, nhead=1, dim_feedforward=1024)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=2)

        self.cls_layer = nn.Sequential(
            nn.Linear(200, 10),
            nn.Sigmoid()
        )

        self.layer4 = nn.Linear(200, 400)
        self.layer5_1 = nn.Linear(400, 500)
        self.layer5_2 = nn.Linear(400, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(500, input_B)
        self.drop = 0.5

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init('xavier'))
        self.flatten = nn.Flatten()
        self.recon_criterion = nn.BCELoss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, 784)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, 784)).unsqueeze(1)
        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        x = x.mean(dim=0)
        y = self.cls_layer(x)

        return y

    def decoder(self, latent):
        x = F.dropout(F.relu(self.layer4(latent)), self.drop)

        out1 = F.relu(self.layer5_1(x))
        out1 = self.sigmoid(self.layer6_1(out1))
        out2 = F.relu(self.layer5_2(x))
        out2 = self.sigmoid(self.layer6_2(out2))
        return out1.view(-1, 1, 28, 28), out2.view(-1, 1, 28, 28)

    def get_loss(self, Xs, labels=None):
        if labels is not None:
            y = self(Xs)

            cls_loss = self.cls_criterion(y, labels)
            _, preds = torch.max(y, 1)
            batch_acc = torch.sum(preds == labels)

            #             latent = self.test_commonZ(Xs)
            #             recon1, recon2 = self.decoder(latent)
            #             recon_loss = 0.3 * self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
            #             0.7 * self.recon_criterion(recon1, Xs[0]).mean(0).sum()
            #             loss = 0.5 * recon_loss + 0.5 * cls_loss
            return cls_loss
        else:
            latent = self.test_commonZ(Xs)
            recon1, recon2 = self.decoder(latent)
            recon_loss = 0.3 * self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
                         0.7 * self.recon_criterion(recon1, Xs[0]).mean(0).sum()
            return recon_loss

    def test_commonZ(self, Xs):
        # x1, x2 = Xs
        # x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 784))), self.drop)
        # x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 784))), self.drop)
        #
        # latent = self.extract_layers(torch.cat((x1, x2), 1))

        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, 784)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, 784)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        latent = x.mean(dim=0)
        return latent

    def get_cls_optimizer(self):
        self.cls_optimizer = torch.optim.SGD(chain(self.encoder1.parameters(), self.encoder2.parameters(),
                                                   self.extract_layers.parameters(), self.cls_layer.parameters()),
                                             lr=1e-3,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        self.cls_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.cls_optimizer, milestones=[150, 300, 600],
                                                                  gamma=0.1)
        return self.cls_optimizer

    def get_recon_optimizer(self):
        self.recon_optimizer = torch.optim.SGD(self.parameters(),
                                               lr=2 * 1e-3,
                                               momentum=0.9,
                                               weight_decay=5e-4,
                                               )
        self.recon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.recon_optimizer, milestones=[150, 300, 600],
                                                                    gamma=0.1)
        return self.recon_optimizer


class Net4HW(BaseNet):
    def __init__(self, input_A, input_B, nz):
        super().__init__()
        self.input_A = input_A
        self.input_B = input_B
        self.nz = nz
        self.encoder1 = nn.Sequential(
            nn.Linear(input_A, 150, bias=False),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_B, 50, bias=True),
            nn.Tanh()
        )

        # Add Transformer.
        self.trans_enc = nn.TransformerEncoderLayer(d_model=self.nz, nhead=1, dim_feedforward=1024)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

        self.cls_layer = nn.Sequential(
            nn.Linear(self.nz, 10),
            nn.Sigmoid()
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(self.nz, 200),
            nn.LeakyReLU(True),
            nn.Linear(200, self.input_A),
            nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(self.nz, self.input_B),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init('kaiming'))
        self.flatten = nn.Flatten()
        self.recon_criterion = nn.BCELoss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        x = x.mean(dim=0)
        y = self.cls_layer(x)

        return y

    def decoder(self, latent):

        out1 = self.decoder1(latent)
        out2 = self.decoder2(latent)
        return out1.view(-1, self.input_A), out2.view(-1, self.input_B)

    def get_loss(self, Xs, labels=None):
        if labels is not None:
            y = self(Xs)
            cls_loss = self.cls_criterion(y, labels)
            return cls_loss
        else:
            latent = self.test_commonZ(Xs)
            recon1, recon2 = self.decoder(latent)
            recon_loss = self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
                         self.recon_criterion(recon1, Xs[0]).mean(0).sum()
            return recon_loss

    def test_commonZ(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1.view(-1, self.input_A)).unsqueeze(1)
        x2 = self.encoder2(x2.view(-1, self.input_B)).unsqueeze(1)

        x = self.extract_layers(torch.cat((x1, x2), 2))
        # out: (batch size, length, d_model)
        x = x.transpose(0, 1)
        # mean pooling
        latent = x.mean(dim=0)
        return latent

    def get_cls_optimizer(self):
        self.cls_optimizer = torch.optim.SGD(chain(self.encoder1.parameters(), self.encoder2.parameters(),
                                                   self.extract_layers.parameters(), self.cls_layer.parameters()),
                                             lr=1e-3,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        self.cls_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.cls_optimizer, milestones=[40, 150, 270],
                                                                  gamma=0.1)
        return self.cls_optimizer

    def get_recon_optimizer(self):
        self.recon_optimizer = torch.optim.SGD(self.parameters(),
                                               lr=2 * 1e-3,
                                               momentum=0.9,
                                               weight_decay=5e-4,
                                               )
        self.recon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.recon_optimizer, milestones=[40, 150, 270],
                                                                    gamma=0.1)
        return self.recon_optimizer