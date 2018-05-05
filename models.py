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


        self.conv1 = nn.Conv2d(1, 32, 3) # 224 - 3 + 1 = 222
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2) # (111, 111)
        self.pool1_drop = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3) # 111 - 3 + 1 = 109
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2) # (54, 54)
        self.pool2_drop = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3) # 54 - 3 + 1 = 52
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2) # (26, 26)
        self.pool3_drop = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 5) # 26 - 5 + 1 = 22
        self.conv4_bn = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2) # (11, 11)
        self.pool4_drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 11 * 11, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1000, 136)


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.pool1(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool1_drop(x)
        x = self.pool2(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool2_drop(x)
        x = self.pool3(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.pool3_drop(x)
        x = self.pool4(self.conv4_bn(F.relu(self.conv4(x))))
        x = self.pool4_drop(x)

        x = x.view(x.size(0), -1)

        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc1_drop(x)

        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc2_drop(x)

        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
