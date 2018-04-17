import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model import ClassificationNet

# train_set['data'][0]['features'].shape = (10,512)
train_set = pickle.load(open("./data/annotated_train_set.p", "rb"),encoding='latin1' )
test_set = pickle.load(open("./data/randomized_annotated_test_set_no_name_no_num.p", "rb"), encoding='latin1')
net = ClassificationNet()
train_label = train_set['data'][0]['class_name'] = 'pour'

epoch_num = 20
class_num = 51
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
def main():
    print("Start training")
    net.train()
    label_batch = None
    output_batch = None
    for epoch in range(epoch_num):
        np.random.shuffle(train_set['data'])
        for i, data_point in enumerate(train_set['data'][:-1000:]):
            # input_dict.shape = N*5120 (10*512)
            input = data_point['features'].flatten()
            label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor)

            label_np = np.asarray([label],dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor)

            outputs = net(input_var)
            # output_dict.shape = N*51
            if label_batch is None or output_batch is None:
                output_batch = outputs.view(1,51)
                # label_batch = label_var.view(1,51)
                label_batch = label_var
            else:
                output_batch = torch.cat((output_batch, outputs.view(1,51)),0)
                # label_batch = torch.cat((label_batch, label_var.view(1,51)),0)
                label_batch = torch.cat((label_batch,label_var),0)

            if i%32 is 0 and i is not 0:
                loss = criterion(output_batch, label_batch)
                # print("Iter: {}, loss: {}".format(i, loss))
                loss.backward()
                optimizer.step()

                label_batch = None
                output_batch = None


            if i%1000 is 0 and i is not 0:
                test_train(train_set['data'][-1000::],epoch,i)


def test_train(test_train_data,epoch,iter):
    print("test_train")
    net.eval()
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
            input_var = Variable(input_tensor)

            label_np = np.asarray([label],dtype=np.long)
            label_tensor = torch.from_numpy(label_np)
            label_var = Variable(label_tensor)

            outputs = net(input_var)


            # output_dict.shape = N*51
            if label_batch is None or output_batch is None:
                output_batch = outputs.view(1,51)
                label_batch = label_var
            else:
                output_batch = torch.cat((output_batch, outputs.view(1,51)),0)
                label_batch = torch.cat((label_batch,label_var),0)

        # if i%eval_num is 0 and i is not 0:
        pred = output_batch.data.max(1, keepdim=True)[1]
        correct += pred.eq(label_batch.data.view_as(pred)).long().cpu().sum()
        loss = criterion(output_batch, label_batch)
        print("Testing Epoch: {} Iter: {}, accuracy: {}/{}".format(epoch,iter,correct,len(test_train_data)))
        label_batch = None
        output_batch = None
        correct = 0

def test():
    net.eval()
    print("Start testing")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    correct = 0
    label_batch = None
    output_batch = None
    for echo in range(epoch_num):
        np.random.shuffle(test_set['data'])
        for i, data_point in enumerate(test_set['data']):
            # input_dict.shape = N*5120 (10*512)
            input = data_point['features'].flatten()
            # label = data_point['class_num']

            input_tensor = torch.from_numpy(input)
            input_var = Variable(input_tensor)

            # label_np = np.asarray([label],dtype=np.long)
            # label_tensor = torch.from_numpy(label_np)
            # label_var = Variable(label_tensor)

            outputs = net(input_var)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(outputs.data.view_as(pred)).long().cpu().sum()

            # output_dict.shape = N*51
            if label_batch is None or output_batch is None:
                output_batch = outputs.view(1,51)
                # label_batch = label_var.view(1,51)
                # label_batch = label_var
            else:
                output_batch = torch.cat((output_batch, outputs.view(1,51)),0)
                # label_batch = torch.cat((label_batch, label_var.view(1,51)),0)
                # label_batch = torch.cat((label_batch,label_var),0)

            if i%100 is 0 and i is not 0:
                loss = criterion(output_batch, label_batch)
                print("Testing Iter: {}, loss: {} accuracy: {}".format(i, loss, correct))
                label_batch = None
                output_batch = None

if __name__ == "__main__":
    main()