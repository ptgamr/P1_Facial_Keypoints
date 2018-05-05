## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # MODEL 3
        # self.conv1 = nn.Conv2d(1, 256, 5) # (220,220)
        # self.pool1 = nn.MaxPool2d(2) # (110, 110)
        # self.conv2 = nn.Conv2d(256, 128, 5) # (106, 106)
        # self.pool2 = nn.MaxPool2d(2) # (53, 53)
        # self.conv3 = nn.Conv2d(128, 64, 4) # (50, 50)
        # self.pool3 = nn.MaxPool2d(2) # (25, 25)
        # self.fc1 = nn.Linear(64 * 25 * 25, 1000)
        # self.fc1_drop = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1000, 500)
        # self.fc2_drop = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(500, 136)

        # MODEL 4
        # self.conv1 = nn.Conv2d(1, 32, 5) # (220,220)
        # self.pool1 = nn.MaxPool2d(2) # (110, 110)
        # self.conv2 = nn.Conv2d(32, 64, 5) # (106, 106)
        # self.pool2 = nn.MaxPool2d(2) # (53, 53)
        # self.conv3 = nn.Conv2d(64, 128, 4) # (50, 50)
        # self.pool3 = nn.MaxPool2d(2) # (25, 25)
        # self.conv4 = nn.Conv2d(128, 256, 3) # (23, 23)
        # self.pool4 = nn.MaxPool2d(2) # (11, 11)
        # self.fc1 = nn.Linear(256 * 11 * 11, 1000)
        # self.fc1_drop = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1000, 1000)
        # self.fc2_drop = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(1000, 136)

        # MODEL 5
        # self.conv1 = nn.Conv2d(1, 32, 5) # (220,220)
        # self.pool1 = nn.MaxPool2d(2) # (110, 110)
        # self.pool1_drop = nn.Dropout(0.1)
        # self.conv2 = nn.Conv2d(32, 64, 5) # (106, 106)
        # self.pool2 = nn.MaxPool2d(2) # (53, 53)
        # self.pool2_drop = nn.Dropout(0.2)
        # self.conv3 = nn.Conv2d(64, 128, 4) # (50, 50)
        # self.pool3 = nn.MaxPool2d(2) # (25, 25)
        # self.pool3_drop = nn.Dropout(0.3)
        # self.conv4 = nn.Conv2d(128, 256, 3) # (23, 23)
        # self.pool4 = nn.MaxPool2d(2) # (11, 11)
        # self.pool4_drop = nn.Dropout(0.4)
        # self.fc1 = nn.Linear(256 * 11 * 11, 1000)
        # self.fc1_drop = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1000, 1000)
        # self.fc2_drop = nn.Dropout(0.6)
        # self.fc3 = nn.Linear(1000, 136)

        # MODEL 7
        self.conv1 = nn.Conv2d(1, 32, 5) # (220,220)
        self.pool1 = nn.MaxPool2d(2) # (110, 110)
        self.pool1_drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(32, 64, 5) # (106, 106)
        self.pool2 = nn.MaxPool2d(2) # (53, 53)
        self.pool2_drop = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(64, 128, 4) # (50, 50)
        self.pool3 = nn.MaxPool2d(2) # (25, 25)
        self.pool3_drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 25 * 25, 500)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 500)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 136)


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1_drop(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2_drop(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool3_drop(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
