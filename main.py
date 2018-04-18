import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from model import ClassificationNet, RNN

epoch_num = 100
class_num = 51
criterion = nn.CrossEntropyLoss()

# train_set['data'][0]['features'].shape = (10,512)
train_set = pickle.load(open("./data/annotated_train_set.p", "rb"), encoding='latin1')
test_set = pickle.load(open("./data/randomized_annotated_test_set_no_name_no_num.p", "rb"), encoding='latin1')
fcn_net = ClassificationNet().cuda()

input_size = 512
hidden_size = 128
output_size = 51
rnn_net = RNN(input_size, hidden_size, output_size).cuda()
# train_label = train_set['data'][0]['class_name'] = 'pour'
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(fcn_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


def train_rnn():
    print("Start training")
    rnn_net.train()
    rnn_net.zero_grad()
    hidden = rnn_net.initHidden().cuda()
    label_batch = None
    output_batch = None

    for epoch in range(epoch_num):
        np.random.shuffle(train_set['data'])
        for i, data_point in enumerate(train_set['data'][0:3200:]):

            # input.shape = 10*512
            input = data_point['features']
            label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor).cuda()

            label_np = np.asarray([label], dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor).cuda()

            # train = data_utils.TensorDataset(input_var, label_var)
            # train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
            for frame_idx, frame in enumerate(input_var):
                output, hidden = rnn_net(frame.view(1,-1), hidden)

            loss = criterion(output, label_var)
            loss.backward(retain_graph=True)

            if i % 100 is 0 and i is not 0:
                print("Evaluating")
                test_rnn_train(train_set['data'][3200::], epoch, i, hidden)


def train_fcn():
    print("Start training")
    fcn_net.train()
    label_batch = None
    output_batch = None
    for epoch in range(epoch_num):
        np.random.shuffle(train_set['data'])
        for i, data_point in enumerate(train_set['data'][0:3200:]):
            # input_dict.shape = N*5120 (10*512)
            input = data_point['features'].flatten()
            label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor).cuda()

            label_np = np.asarray([label], dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor).cuda()

            outputs = fcn_net(input_var)
            # output_dict.shape = N*51
            if label_batch is None or output_batch is None:
                output_batch = outputs.view(1, 51)
                # label_batch = label_var.view(1,51)
                label_batch = label_var
            else:
                output_batch = torch.cat((output_batch, outputs.view(1, 51)), 0)
                # label_batch = torch.cat((label_batch, label_var.view(1,51)),0)
                label_batch = torch.cat((label_batch, label_var), 0)

            if i % 32 is 0 and i is not 0:
                loss = criterion(output_batch, label_batch)
                # print("Iter: {}, loss: {}".format(i, loss))
                loss.backward()
                optimizer.step()

                label_batch = None
                output_batch = None

            if i % 1000 is 0 and i is not 0:
                test_fcn_train(train_set['data'][3200::], epoch, i)


def test_fcn_train(test_train_data, epoch, iter):
    print("test_train")
    fcn_net.eval()
    correct = 0
    eval_num = 100
    label_batch = None
    output_batch = None
    for echo in range(1):
        for i, data_point in enumerate(test_train_data):
            # input_dict.shape = N*5120 (10*512)
            input = data_point['features'].flatten()
            label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor).cuda()

            label_np = np.asarray([label], dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor).cuda()

            outputs = fcn_net(input_var)

            # output_dict.shape = N*51
            if label_batch is None or output_batch is None:
                output_batch = outputs.view(1, 51)
                label_batch = label_var
            else:
                output_batch = torch.cat((output_batch, outputs.view(1, 51)), 0)
                label_batch = torch.cat((label_batch, label_var), 0)

            if i%eval_num is 0 and i is not 0:
                pred = output_batch.data.max(1, keepdim=True)[1]
                correct += pred.eq(label_batch.data.view_as(pred)).long().cpu().sum()
                loss = criterion(output_batch, label_batch)
                print("Testing Epoch: {} Iter: {}, correct: {} ,accuracy: {}".format(epoch, iter, correct,
                                                                                     (correct / len(test_train_data)) * 100))
                label_batch = None
                output_batch = None
                correct = 0


def test_rnn_train(test_train_data, epoch, iter, hidden):
    print("test_rnn_train")
    rnn_net.eval()
    correct = 0
    eval_num = 100
    label_batch = None
    output_batch = None
    for echo in range(1):
        for i, data_point in enumerate(test_train_data):
            # input_dict.shape = N*5120 (10*512)
            input = data_point['features']
            label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor).cuda()

            label_np = np.asarray([label], dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor).cuda()

            for frame_idx, frame in enumerate(input_var):
                output, hidden = rnn_net(frame.view(1, -1), hidden)

            # if i%eval_num is 0 and i is not 0:
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label_var.data.view_as(pred)).long().cpu().sum()
        # loss = criterion(output_batch, label_var)
        print("Testing Epoch: {} Iter: {}, correct: {} ,accuracy: {}".format(epoch, iter, correct,
                                                                             (correct / len(test_train_data)) * 100))
        label_batch = None
        output_batch = None
        correct = 0


if __name__ == "__main__":
    train_fcn()
    # train_rnn()