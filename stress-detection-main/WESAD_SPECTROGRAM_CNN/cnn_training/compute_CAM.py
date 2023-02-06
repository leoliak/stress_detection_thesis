import os
import numpy as np
from datetime import datetime
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import matplotlib.pyplot as plt
import sklearn.metrics as skm

import albumentations as A
from PIL import Image

import math
import cv2

from torchvision import datasets, transforms
from sklearn import metrics
from torchvision import models
import pdb

sample_weight=None
normalize=None
display_labels=None
include_values=True
xticks_rotation='horizontal'
values_format=None
ax=None


def visualize_comfusion_matrix(y_true, y_pred, class_names, print_name):
    # Plot non-normalized confusion matrix
    titles_options = [print_name + " Confusion matrix"]
    for title in titles_options:
        cm = metrics.confusion_matrix(y_true, y_pred)

        dis = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=display_labels)
        disp = dis.plot(include_values=include_values,
                            cmap=plt.cm.Blues, ax=ax, xticks_rotation=xticks_rotation,
                            values_format=values_format)
        disp.ax_.set_title(title)
        plt.savefig(RESULTS_FOLDER + "/" + print_name.replace(' ', '_') + ".png")
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






def loader():
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
    trainloader = DataLoader(dataset=image_datasets_train, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    testloader = DataLoader(dataset=image_datasets_eval, 
                            batch_size=1, 
                            shuffle=False,
                            pin_memory=True,
                            num_workers=0)
    return trainloader, testloader




def ComputeCAMs(net, img_tensor, epoch=0):
    # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
    net.cpu()
    net.eval()
    finalconv_name = 'layer4'

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        # for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        # output_cam.append(cv2.resize(cam_img, size_upsample))
        return cam_img

    cam_img = []
    heatmap = []
    result = []
 
    logit = net(img_tensor)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = []
    for i in range(0,2):
        CAM = returnCAM(features_blobs[-1], weight_softmax, [idx[0]])
        CAMs.append(CAM)
    
    return CAMs
    


def train():
    steps = 0
    running_loss = 0
    print_every = 10
    best_accuracy = -1.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(epochs):
        accuracy_tr = 0
        for inputs, labels, _ in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs.to(device))
            loss = criterion(logps, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_tr += torch.mean(equals.type(torch.FloatTensor)).item()
        
        scheduler.step()
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels, _ in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(running_loss/len(trainloader))
        train_accs.append(accuracy_tr/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        test_accs.append(accuracy/len(testloader)) 
        print(test_accs)
        
        print(f"Epoch {epoch+1}/{epochs} "
                f"Train loss: {running_loss/len(trainloader):.5f} "
                f"Test loss: {test_loss/len(testloader):.5f} "
                f"Train accuracy: {accuracy_tr/len(trainloader):.5f} "
                f"Test accuracy: {accuracy/len(testloader):.5f} "
                f"Best accuracy: {100*best_accuracy:.3f}")

        accc = accuracy/len(testloader)
        if accc > best_accuracy:
            best_accuracy = accc
            best_name = RESULTS_FOLDER +'/model_best_epoch_{}.pth'.format(epoch)
            torch.save(model, RESULTS_FOLDER +'/model_best_epoch_{}.pth'.format(epoch))
        if best_accuracy>0.95: 
            print("best_found")
            break
        running_loss = 0
        model.train()
    
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.ylim(0, 2)
    plt.legend(frameon=False)
    plt.savefig(RESULTS_FOLDER + "/loss_plot.png")
    plt.close()

    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.ylim(0, 1.2)
    plt.legend(frameon=False)
    plt.savefig(RESULTS_FOLDER + "/acc_plot.png")
    plt.close()

    return best_name



def evaluation(bn):
    print("Final evaluation is online, please wait for results...")
    data_dir = DATASET_FOLDER + '/eval'
    test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model=torch.load(bn)

    classes = ["no_stress", "stress"]

    def predict_image(image):
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device) 
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        return index


    lbl = []
    gt = []
    print(device)
    model.eval()
    model.to(device)

    with torch.no_grad():
        cc = 0
        for inputs, labels, image in testloader:
            print("Process image " + str(cc))
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            logps = F.sigmoid(logps.double())
            idx = logps.argmax(dim=1).detach().cpu().numpy()
            gtt = labels.detach().cpu().numpy()
            lbl.append(idx[0])
            gt.append(gtt[0])

            CAMs = ComputeCAMs(model, inputs)
            # render the CAM and output
            heatmap = []
            result = []
            cam_img = []
            image = image.squeeze().numpy()
            print(RES_CAM)
            for i in range(0,2):
                height, width, _ = image.shape
                cam_img1 = cv2.resize(CAMs[i],(width, height))
                heatmap1 = cv2.applyColorMap(cam_img1, cv2.COLORMAP_JET)
                result1 = heatmap1 * 0.35 + image * 0.6
                cv2.imwrite(RES_CAM + "/image_{}_{}.png".format(i, cc), result1)
            cc += 1
    lbl1 = np.array(lbl)
    gt1 = np.array(gt)
    cm = metrics.confusion_matrix(gt1, lbl1)
    print(cm.ravel())
    print(metrics.classification_report(gt1, lbl1))

    visualize_comfusion_matrix(gt1, lbl1, ["no stress", "stress"], "image_classifier")

    print("Metrics")
    tn, fp, fn, tp = cm.ravel()
    print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')




## TRAINING HYPERPARAMETERS
LEARNING_RATE = 0.00082
BATCH_SIZE = 16
epochs = 100
AUGS = True

## MODEL PARAMETERS
IMG_SIZE = 224
CHANNEL_IN = 3
N_CLASSES = 2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(device)
# device = "cpu"
LABELS = ["stress", "no_stress"]

steps = 0
running_loss = 0
print_every = 10

DATASET_FOLDER = "/mnt/sdb/thesis/WESAD_experiments/ECG/datasets/dataset_2_extended"

## Model loader
# model = CNN_resnet18(N_CLASSES, CHANNEL_IN).to(device)

model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
                nn.Linear(512, 2),
                nn.Softmax(dim=1))

EXP_NAME = "ecg_resnet18_dataset_2_v2_allgrad_2fc_dummy"
model.to(device)
## Training parameters
trainloader, testloader = loader()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)


## FOLDERS
RESULTS_FOLDER = "results/" + EXP_NAME
os.makedirs(RESULTS_FOLDER, exist_ok=True)

## FOR CAM
RES_CAM = RESULTS_FOLDER + "/CAMS"
os.makedirs(RES_CAM, exist_ok=True)

train_fl = True
eval_fl = True

if train_fl:
    bn = train()
if eval_fl:
    if not train_fl:
        bn = RESULTS_FOLDER + "/model_best_epoch_10.pth" 
    evaluation(bn)

