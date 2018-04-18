from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        class_num = 51
        self.fc1 = nn.Linear(512*10,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # combined 57+128=185
        hidden = self.i2h(combined) # hidden = (1,128)
        output = self.i2o(combined) # output = (1,18)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()