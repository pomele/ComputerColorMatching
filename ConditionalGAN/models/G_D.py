import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
# from IPython import embed


class Generator(nn.Module):
    def __init__(self, N_IDEAS, SOURCE_COLOR_NUM):
        super(Generator, self).__init__()
        self.g = nn.Sequential(  # Generator
            nn.Linear(N_IDEAS + 31, 512),  # random ideas (could from normal distribution) + class label
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),  # random ideas (could from normal distribution) + class label
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1536),  # random ideas (could from normal distribution) + class label
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),  # random ideas (could from normal distribution) + class label
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # random ideas (could from normal distribution) + class label
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, SOURCE_COLOR_NUM),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        """
        :param images:
        :return:
        """
        output = self.g.forward(input_data)
        return output


class Discriminator(nn.Module):
    def __init__(self, SOURCE_COLOR_NUM):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(  # Discriminator
            nn.Linear(SOURCE_COLOR_NUM + 31, 64),  # receive art work either from the K-M or a newbie like G with label
            nn.ReLU(),
            nn.Linear(64, 128),  # receive art work either from the K-M or a newbie like G with label
            nn.ReLU(),
            nn.Linear(128, 64),  # receive art work either from the K-M or a newbie like G with label
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input_data):
        output = self.d.forward(input_data)
        return output
