# -*- coding: utf-8 -*-

from PIL import Image
import torch
from torch.utils import data
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch import nn, optim
import torchvision.models as models


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#From Stanford tutorial
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")

class MTurkTrain(Dataset):
  def __init__(self,csv_file):
    self.data_frame = pd.read_csv(csv_file)
    self.img_dir = "/global/scratch/oafolabi/data/mturkCSVs/m-turk"

  def __len__(self):
    return self.data_frame.shape[0]

  def __getitem__(self,idx):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_label_pair = self.data_frame.iloc[idx]
    img_name = img_label_pair[0]
    img = Image.open(self.img_dir +'/'+ img_name)
    img = transform(img)
    label = img_label_pair[1]
    return img,label

params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 0}

train_dataset = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/train_data.csv")
training_generator = data.DataLoader(train_dataset, **params)

validation_set = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/val_data.csv")
validation_generator = data.DataLoader(validation_set, **params)

"""**Arbaaz's Bootlegged Model**"""

import os
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 1280, 1280)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 640, 640)
            nn.ReLU(),
            nn.BatchNorm2d(16),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 640, 640)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 640, 640)
            nn.Conv2d(
                in_channels=16,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 640, 640)
            nn.ReLU(),
            nn.BatchNorm2d(32),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 320, 320)
        )

        self.conv3 = nn.Sequential(         # input shape (32, 320, 320)
            nn.Conv2d(
                in_channels=32,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (64, 320, 320)
            nn.ReLU(),
            nn.BatchNorm2d(64),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (64, 160, 160)
        )

        self.conv4 = nn.Sequential(         # input shape (64, 160, 160)
            nn.Conv2d(
                in_channels=64,              # input height
                out_channels=128,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (128, 160, 160)
            nn.ReLU(),
            nn.BatchNorm2d(128),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (128, 80, 80)
        )

        self.conv5 = nn.Sequential(         # input shape (128, 80, 80)
            nn.Conv2d(
                in_channels=128,              # input height
                out_channels=256,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (256, 80, 80)
            nn.ReLU(),
            nn.BatchNorm2d(256),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (256, 40, 40)
        )

        self.conv6 = nn.Sequential(         # input shape (256, 40, 40)
            nn.Conv2d(
                in_channels=256,              # input height
                out_channels=512,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (512, 40, 40)
            nn.ReLU(),
            nn.BatchNorm2d(512),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (512, 20, 20)
        )

        self.conv7 = nn.Sequential(         # input shape (512, 20, 20)
            nn.Conv2d(
                in_channels=512,              # input height
                out_channels=1024,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (1024, 20, 20)
            nn.ReLU(),
            nn.BatchNorm2d(1024),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (1024, 10, 10)
        )

        self.conv8 = nn.Sequential(         # input shape (1024, 10, 10)
            nn.Conv2d(
                in_channels=1024,              # input height
                out_channels=2048,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (2048, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(2048),                   # activation
            nn.MaxPool2d(kernel_size=2)    # choose max value in 2x2 area, output shape (2048, 5, 5)
        )

        self.post_flatten = nn.Sequential(
            nn.Linear(512000,16),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 2),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = torch.flatten(x)
        x = x.unsqueeze(0)
        x = self.post_flatten(x)
        return x


# cnn = CNN()
# cnn.cuda()

"""**Training**"""

#From
# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/

model = CNN()
model.to(device)

max_epochs = 1

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

start_ts = time.time()
model.train()
batches = params['batch_size']

for epoch in range(max_epochs):
    print("EPOCH: " + str(epoch))
    total_loss = 0
  #Training
    for idx, data in enumerate(training_generator):
        X, y = data[0].to(device), data[1].to(device)
        model.zero_grad()
        outputs = model(X)
        output_array = torch.argmax(outputs.data)
        print(output_array)
        print("     on to loss")
        loss = loss_function(outputs, y)
        loss.backward()
        print("     backprop")
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        print("     Loss: {:.4f}".format(total_loss/(idx+1)))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""**Save Model**"""

path = "8cnn"
torch.save(model.state_dict(), path)

"""**Load Pre-saved Model**"""

#model_save_name = 'maxed_cnn.pt'
#path = F"/content/gdrive/My Drive/VRResearch/{model_save_name}"
#model.load_state_dict(torch.load(path))

"""**Validation**"""

# Validation

model.eval()
with torch.set_grad_enabled(False):
  val_wrong = 0
  for i, data in enumerate(validation_generator):
    print("Current: " + str(i) + " / " )
    # Transfer to GPU
    X, y = data[0].to(device), data[1].to(device)
     # Model computations
    outputs = model(X)
    predicted_classes = torch.max(outputs, 1)[1]
    prediction_lst = predicted_classes.tolist()
    val_wrong += sum([1 if prediction_lst[i] != y[i] else 0 for i in range(len(prediction_lst))])


print(1 - (val_wrong / 387))
