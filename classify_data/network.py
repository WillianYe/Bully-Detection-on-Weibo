import torch.nn as nn
from classify_data.config import *

class nnNet(nn.Module):
    def __init__(self):
        super(nnNet, self).__init__()
        self.fc1 = nn.Linear(EMBEDDING_SIZE, 96)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(96, 32)
        self.fc3 = nn.Linear(32, CLASS_NUMBER)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax()


    def forward(self, input, is_training=False):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        if not is_training:
            output = self.softmax(output)
        return output


class cnnNet(nn.Module):
    def __init__(self):
        super(cnnNet, self).__init__()  # 3*10*10
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3))  # 6*8*8
        self.batchnormal1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 6*4*4
        self.conv2 = nn.Conv2d(6, 48, kernel_size=(3, 3))  # 48*2*2
        self.batchnormal2 = nn.BatchNorm2d(48)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 48*1*1
        self.flatten = nn.Flatten()  # 16
        self.fc = nn.Linear(48, 120)  # 10
        self.fc2 = nn.Linear(120, CLASS_NUMBER)
        self.dropout = nn.Dropout2d(0.1)
        self.softmax = nn.Softmax()

    def forward(self, input, is_training=False):
        output = self.conv1(input)
        output = self.batchnormal1(output)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.batchnormal2(output)
        output = self.relu(output)
        output = self.maxpool2(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.relu(output)
        output = self.fc2(output)
        if not is_training:
            output = self.softmax(output)
        return output


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn1 = nn.LSTM(EMBEDDING_SIZE, 96 ,num_layers=3)
        self.rnn2 = nn.LSTM(96, 32, num_layers=3)
        self.rnn3 = nn.LSTM(32, CLASS_NUMBER, num_layers=3)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, input, is_training=False):
        input = input.unsqueeze(dim=0)
        output, _ = self.rnn1(input)
        output, _ = self.rnn2(output)
        output, _ = self.rnn3(output)
        output = output.squeeze(dim=0)
        output = self.dropout(output)
        if not is_training:
            output = self.softmax(output)

        return output


