import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN_resnet34(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_resnet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out



class CNN_alexnet(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_alexnet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out


class CNN_vgg16(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out



class CNN_inception_v3(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_inception_v3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out




class CNN_mobilenet_v2(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_mobilenet_v2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out



class CNN_resnet18(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN_resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # for self.param in self.model.parameters():
        #    self. param.requires_grad = False

        # self.fc = nn.Sequential(
        #     nn.Linear(1000, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_channels),
        #     nn.Softmax(dim=1))

        self.fc = nn.Sequential(
                nn.Linear(1000, 512),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(512, n_channels),
                nn.Softmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out
