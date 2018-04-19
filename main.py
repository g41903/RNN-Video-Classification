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
optimizer = optim.Adam(fcn_net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


def fcn_data_loader(train_data):
    data_frames = []
    for data_idx, data in enumerate(train_data):
        data_label = data['class_num']
        for frame_idx, frame in enumerate(data['features']):
            data_frame = (frame, data_label)
            data_frames.append(data_frame)
    return data_frames


def train_rnn():
    print("Start training")
    rnn_net.train()
    rnn_net.zero_grad()
    hidden = rnn_net.initHidden().cuda()
    label_batch = None
    output_batch = None
    np.random.shuffle(train_set['data'])
    for epoch in range(epoch_num):

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
    np.random.shuffle(train_set['data'])
    train_data = train_set['data'][:3000:]
    valid_data = train_set['data'][3000::]
    # np.random.shuffle(valid_data)
    data_frames = fcn_data_loader(train_data)
    for epoch in range(epoch_num):
        np.random.shuffle(data_frames)

        for i, (frame, label) in enumerate(data_frames[:35000:]):
            # input = data_point['features']
            # label = data_point['class_num']

            input_tensor = torch.from_numpy(frame)
            input_var = Variable(input_tensor).cuda()

            label_np = np.asarray([label], dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor).cuda()
            # if input_var.data.shape[0] != 512:
            #     print("Shape is not correct")
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

            if i % 100 is 0 and i is not 0:
                optimizer.zero_grad()
                loss = criterion(output_batch, label_batch)
                # print("Iter: {}, loss: {}".format(i, loss))
                loss.backward()
                optimizer.step()

                label_batch = None
                output_batch = None

            if i % 1000 is 0 and i is not 0:
                # test_fcn_train(data_frames[35000::], epoch, i)
                test_fcn_train(valid_data,epoch,i)


def test_fcn_train(test_train_data, train_epoch, iter):
    print("test_train")
    fcn_net.eval()
    correct = 0
    eval_num = 100
    pred_outputs = []
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

            for frame_idx, input_frame_var in enumerate(input_var):
                pred_output_var = fcn_net(input_frame_var)
                pred_num = pred_output_var.data.max(0, keepdim=True)[1].cpu().numpy()[0]
                pred_outputs.append(pred_num)


            # get the most frequent prediction
            bincount = np.bincount(pred_outputs)
            output_idx = np.argmax(bincount)
            # output_zeros = np.zeros(class_num)
            # output_zeros[output_idx] = 1
            # output_tensor = torch.from_numpy(output_zeros)
            # outputs_var = Variable(output_tensor).cuda()

            pred_outputs = []

            if output_idx == label:
                correct +=1

        print("Testing Epoch: {} Iter: {}, correct: {} ,accuracy: {}".format(train_epoch, iter, correct,
                                                                             (correct / len(test_train_data)) * 100))
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