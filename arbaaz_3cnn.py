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
    if(csv_file == "/global/scratch/oafolabi/data/mturkCSVs/train_data.csv"):
        self.data_frame = self.data_frame.iloc[3:]

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

print("SIZE")
print(train_dataset.__len__())

validation_set = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/val_data.csv")
validation_generator = data.DataLoader(validation_set, **params)

"""**Arbaaz's Bootlegged Model**"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(16),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=16,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(32),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv3 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=32,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(64),                   # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.post_flatten = nn.Sequential(
            nn.Linear(4096, 16),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.post_flatten(x)
        return output

# cnn = CNN()
# cnn.cuda()

"""**Training**"""

#From
# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/

model = CNN()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

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
        #print(outputs.data)
        print("     on to loss")
        loss = loss_function(outputs, y)
        loss.backward()
        print("     backprop")
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        if(idx % 30 == 0):
            print("EPOCH: "  + str(epoch))
        print("     Loss: {:.4f}".format(total_loss/(idx+1)))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""**Save Model**"""

#
# """**Load Pre-saved Model**"""
#
# model.load_state_dict(torch.load("maxed_cnn.pt"))

"""**Validation**"""

# Validation

model.eval()
with torch.set_grad_enabled(False):
  val_wrong = 0
  for i, data in enumerate(validation_generator):
    # Transfer to GPU
    X, y = data[0].to(device), data[1].to(device)
     # Model computations
    outputs = model(X)
    predicted_classes = torch.max(outputs, 1)[1]
    prediction_lst = predicted_classes.tolist()
    val_wrong += sum([1 if prediction_lst[i] != y[i] else 0 for i in range(len(prediction_lst))])

val_acc = 1 - (val_wrong / 387)
if(val_acc > 0.8):
    path = "8cnn"
    torch.save(model.state_dict(), path)
print(1 - (val_wrong / 387))
