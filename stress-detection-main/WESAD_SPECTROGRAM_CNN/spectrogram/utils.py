import os
import numpy as np
from datetime import datetime
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import sklearn.metrics as skm

import albumentations as A
from PIL import Image

import math
import cv2


def plot_confusion_matrix(cm,
                          target_names,
                          best_flag,
                          output_folder,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True
                          ):
    ## Save CM only for best epoch results
    if not best_flag: return
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    pt = output_folder + "/cm.jpg" 
    plt.savefig(pt)
    plt.close()


def plot_losses(train_losses, valid_losses, output_folder):
    '''
    Function for plotting training and validation losses
    '''
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    pt = output_folder + "/losses.jpg" 
    plt.savefig(pt)
    plt.close()
    


# Define the demo dataset
class MyDataset(Dataset):
    def __init__(self, data_folder, class_num, img_size, labels, crop = True, pad = True, augs = True, mode="train", img_side = 300):
        self.data_folder = data_folder
        self.class_num = class_num
        self.labels = labels
        self.IMG_SIZE = img_size
        self.image_list = []
        for fol in os.listdir(self.data_folder):
            for img in os.listdir(self.data_folder + "/" + fol):
                self.image_list.append(os.path.join(self.data_folder, fol, img))
        random.shuffle(self.image_list)
        self.cropping = crop
        self.padding = pad
        self.debug = "debug_img"
        os.makedirs(self.debug, exist_ok=True)
        if mode == "train":
            self.augs = A.Compose([
                            A.Resize(self.IMG_SIZE, self.IMG_SIZE),
                            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ], p=1)
        else:
            self.augs = A.Compose([
                    A.Resize(self.IMG_SIZE, self.IMG_SIZE),
                    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    
    def __len__(self):
        return (len(self.image_list))
    

    def crop(self, image):
        h, w = image.shape[:2]
        image_ = image[6:630, 8:342, :]
        return image_.astype("uint8")
    
    def pad(self, image, max_dim):
        h, w = image.shape[:2]
        nimage = np.zeros((max_dim, max_dim, 3))
        if h >= w:
            ar = w/h
            h_max = max_dim
            w_max = math.floor(max_dim*ar) if int(max_dim*ar) % 2 == 0 else math.ceil(max_dim*ar)
            image_2 = cv2.resize(image, (w_max, h_max))
            init_w = (max_dim - w_max) // 2
            nimage[:, init_w:init_w+w_max, :] = image_2
        else:
            ar = h/w
            w_max = max_dim
            h_max = math.floor(max_dim*ar) if int(max_dim*ar) % 2 == 0 else math.ceil(max_dim*ar)
            image_2 = cv2.resize(image, (w_max, h_max))
            init_h = (max_dim - h_max) // 2
            nimage[init_h:init_h+h_max, :, :] = image_2
        return nimage.astype("uint8")
    
    def __getitem__(self, i):
        image = cv2.imread(self.image_list[i])
        image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_0.jpg", image)
        if self.cropping:
            image = self.crop(image)
        cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_1.jpg", image)
        if self.padding:
            image = self.pad(image, 640)
        cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_2.jpg", image)

        lab = os.path.split(os.path.split(self.image_list[i])[0])[1]
        # print(self.image_list[i], lab)
        image = Image.fromarray(image).convert('RGB')
        image = self.augs(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            
        return torch.tensor(image, dtype=torch.float), torch.tensor(int(self.labels.index(lab)), dtype=torch.long)







# class LeNet5(nn.Module):
#     def __init__(self, n_classes, n_channels):
#         super(LeNet5, self).__init__()
#         self.feature_extractor = nn.Sequential(            
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=64 * 62 * 62 , out_features=64 * 10),
#             nn.ReLU(),
#             nn.Linear(in_features=64 * 10, out_features=64 * 2),
#             nn.ReLU(),
#             nn.Linear(in_features=64 * 2, out_features=n_classes)
#         )

      
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = torch.flatten(x, 1)
#         logits = self.classifier(x)
#         probs = F.softmax(logits, dim=1)
#         return logits, probs
    



class LeNet5(nn.Module):
    def __init__(self, n_classes, n_channels, stn_enable = False):
        super(LeNet5, self).__init__()
        self.stn_activate = stn_enable
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=n_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=331776, out_features=3200),
            nn.ReLU(),
            nn.Linear(in_features=3200, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs



from torchvision import models

class CNN(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for self.param in self.model.parameters():
           self. param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_channels),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out


def get_accuracy(model, data_loader, device, cm = False):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    if cm:
        clsf = []
        gt = []
    correct_pred = 0 
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
            if cm:
                predicted_labels = predicted_labels.cpu().numpy()
                y_true = y_true.cpu().numpy()
                clsf.append(predicted_labels[0])
                gt.append(y_true[0])  
    if cm:
         return correct_pred.float() / n, np.array(clsf), np.array(gt)
    else:
        return correct_pred.float() / n


