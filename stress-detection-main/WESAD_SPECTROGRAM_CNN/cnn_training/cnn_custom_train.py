# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:46:40 2020

@author: leoni
"""

#%% Image Classification

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

from utils import *


########################################################################
##

# Helping functions


def plot_confusion_matrix(cm,
                          target_names,
                          best_flag,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
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
    pt = RESULTS_FOLDER + "/cm.jpg" 
    plt.savefig(pt)
    plt.close()


def plot_losses(train_losses, valid_losses):
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
    pt = RESULTS_FOLDER + "/losses.jpg" 
    plt.savefig(pt)
    plt.close()
    



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



########################################################################
##

########################################################################
##

# Basic functions


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
                        ], p=1)
        else:
            self.augs = A.Compose([
                    A.Resize(self.IMG_SIZE, self.IMG_SIZE),
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
        image_ = cv2.imread(self.image_list[i])
        image_ = cv2.resize(image_, (640, 480), interpolation = cv2.INTER_LINEAR)
        # cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_0.jpg", image_)
        if self.cropping:
            image_ = self.crop(image_)
        cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_1.jpg", image_)
        if self.padding:
            image_ = self.pad(image_, 640)
        cv2.imwrite(self.debug + "/im_" + str(i).zfill(4) + "_2.jpg", image_)

        lab = os.path.split(os.path.split(self.image_list[i])[0])[1]
        # print(self.image_list[i], lab)
        image = image_.copy()
        image = Image.fromarray(image).convert('RGB')
        image = self.augs(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            
        return torch.tensor(image, dtype=torch.float), torch.tensor(int(self.labels.index(lab)), dtype=torch.long), image_



def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0 
    for X, y_true in train_loader:
        optimizer.zero_grad() 
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat= model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
        # Backward pass
        loss.backward()
        optimizer.step()     
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss



def training_loop(model, criterion, scheduler, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    best_dict = {'epoch': 0,
                 'score': 0}
    # set objects for storing metrics
    best_ac = 0
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        scheduler.step()

        # validation
        best_flag = False
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_acc, y_pred, y_true = get_accuracy(model, valid_loader, device=device, cm = True)
            valid_losses.append(valid_loss)
            if valid_acc > best_ac:
                best_flag = True
                save_path = os.path.join(RESULTS_FOLDER, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss
                    }, save_path)
                best_dict["epoch"] = epoch
                best_dict["score"] = valid_acc
                best_ac = valid_acc
                print("New best model saved for validation loss: {} at epoch: {}".format(100 * round(valid_loss, 4), epoch))
        
        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc, y_pred, y_true = get_accuracy(model, valid_loader, device=device, cm = True)
            cm = skm.confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(  
                                cm,
                                LABELS,
                                best_flag,
                                RESULTS_FOLDER,
                                title='Confusion matrix for epoch {}'.format(epoch),
                                cmap=None,
                                normalize=True)
            print("***********"*6)
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}\t'
                  f'lr: {scheduler.get_last_lr()[0]:.5f}')

            print(skm.classification_report(y_true,y_pred))
    plot_losses(train_losses, valid_losses, RESULTS_FOLDER)
    print(f'Best results:'
            f'Epoch: {best_dict["epoch"]}\t'
            f'Valid accuracy: {100 * best_dict["score"]:.2f}\t')
    return model, optimizer, (train_losses, valid_losses)

    




## TRAINING PARAMETERS
LEARNING_RATE = 0.004
BATCH_SIZE = 4
N_EPOCHS = 10
AUGS = False

## MODEL PARAMETERS
IMG_SIZE = 224
CHANNEL_IN = 3
N_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS = ["stress", "no_stress"]

## FOLDERS
EXP_NAME = "experiment_resp_1"
RESULTS_FOLDER = "results/" + EXP_NAME
os.makedirs(RESULTS_FOLDER, exist_ok=True)


DATASET_FOLDER = "/mnt/sdb/thesis/SWELL_2/CNN/datasets/dataset_2"
# Construct datasets
image_datasets_train = MyDataset(DATASET_FOLDER + "/train", 
                                 class_num = N_CLASSES, 
                                 img_size = IMG_SIZE,
                                 labels=LABELS, 
                                 augs=True, 
                                 crop = False,
                                 pad = True,
                                 mode="train", 
                                 img_side = IMG_SIZE)


image_datasets_eval = MyDataset(DATASET_FOLDER + "/eval", 
                                class_num = N_CLASSES,
                                img_size = IMG_SIZE, 
                                labels=LABELS, 
                                augs=True, 
                                crop = False,
                                pad = True,
                                mode="eval", 
                                img_side = IMG_SIZE)


# Define Pytorch Dataloaders
train_loader = DataLoader(dataset=image_datasets_train, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True,
                        pin_memory=True,
                        num_workers=0)

valid_loader = DataLoader(dataset=image_datasets_eval, 
                        batch_size=1, 
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


            
## Model initialization
# model = LeNet5(N_CLASSES, CHANNEL_IN).to(DEVICE)
model = CustomConvNet(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

## Training process start
model, scheduler, _ = training_loop(model, criterion, scheduler, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

print("End of OCR training process")

