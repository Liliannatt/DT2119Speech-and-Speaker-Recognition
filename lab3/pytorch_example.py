
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from loguru import logger
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import time

# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self, input_size, num_cls, hidden_layer_list, activate_func='relu'):
        super(Net, self).__init__()

        if activate_func == 'relu':
            self.activate = nn.ReLU()

        # define the layers of your network here
        self.layer1 = nn.Linear(input_size, hidden_layer_list[0])

        layers = []
        for idx in range(len(hidden_layer_list)-1):
            layers.append(nn.Linear(hidden_layer_list[idx], hidden_layer_list[idx+1]))
            layers.append(self.activate)
        self.hidden_layers = nn.Sequential(*layers)

        self.header_layer = nn.Linear(hidden_layer_list[-1], num_cls)
        # self.final_activate = nn.Sigmoid()

        
    def forward(self, x):
        # define the foward computation from input to output
        x = self.layer1(x)
        x = self.hidden_layers(x)
        x = self.header_layer(x)
        # x = self.final_activate(x)
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def format_time(duration):
    if duration < 60 * 60:
        return f'{int(duration/60):02d}:{int(duration%60):02d}'
    else:
        return f'{int(duration/(60*60)):02d}:{int(duration%(60*60)/60):02d}:{int(duration%(60*60)):02d}'

def calc_accuracy(gt, pred):
    pred = torch.sigmoid(pred)
    pred_cls = pred > 0.5
    correct_pred = (pred_cls == gt).all(dim=1)
    acc = torch.mean(correct_pred, dtype=torch.float32)
    return acc

now = datetime.now()
cur_version = now.strftime("%Y%m%d-%H%M%S")

# Set hyper parameters
# use_dynamic_features = True
use_dynamic_features = False

resume_ckpt = None
# resume_ckpt = 'output/trained-net-20240506-001038.pt'

if use_dynamic_features:
    input_size = 13 * 7
else:
    input_size = 13

stateList = np.load('state_list.npy').tolist()
num_cls = len(stateList)

batch_size = 256
hidden_layer_list = [256, 256, 256, 256]

initial_lr = 0.0001

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# instantiate the network and print the structure
net = Net(input_size=input_size, num_cls=num_cls, hidden_layer_list=hidden_layer_list)
logger.info(net)
logger.info('number of prameters:', count_parameters(net))

net = net.to(device)

if resume_ckpt is not None:
    state_dict = torch.load(resume_ckpt)
    net.load_state_dict(state_dict)
    logger.info(f'Resume checkpoint from {resume_ckpt}')

# define your loss criterion (see https://pytorch.org/docs/stable/nn.html#loss-functions)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# define the optimizer 
optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

# StepLR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# prepare/load the data into tensors 
# train_x = ..., train_y = ..., val_x = ..., val_y = ..., test_x = ..., test_y = ...

prepared_data = np.load('prepared_data.npz')
np_train_x = prepared_data['train_data_x']
np_train_y = prepared_data['train_data_y']

train_x = torch.tensor(np_train_x)
train_y = F.one_hot(torch.tensor(np_train_y, dtype=torch.long), num_classes=num_cls).float()


# create the data loaders for training and validation sets
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# setup logging so that you can follow training using TensorBoard (see https://pytorch.org/docs/stable/tensorboard.html)
writer = SummaryWriter()

# train the network
st = time.time()
last_st = time.time()
num_epochs = 100
log_interval = 100

for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    sub_train_loss = 0.0
    epoch_acc = 0.0
    cnt = 0
    for idx, (inputs, labels) in enumerate(train_loader):

        # move data from cpu to gpu (if available)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # accumulate the training loss
        train_loss += loss.item()

        sub_train_loss += loss.item()

        if idx % log_interval == 0:
            duration = time.time() - st
            acc = calc_accuracy(labels, outputs)
            epoch_acc += (acc * batch_size)
            cnt += batch_size
            logger.info(f'epoch {epoch:03d} / {num_epochs}, step {idx:04d} / {len(train_loader)}, loss {sub_train_loss / log_interval:.5f}, acc {acc*100:.2f}%, total time [{format_time(duration)}], one epoch time {format_time(len(train_loader) / log_interval * (time.time() - last_st))}')
            sub_train_loss = 0.0
            last_st = time.time()

    # update the learning rate
    scheduler.step()

    # # calculate the validation loss
    # net.eval()
    val_loss = 0.0
    # with torch.no_grad():
    #     val_loss = 0.0
    #     for inputs, labels in val_loader:
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()

    # print the epoch loss
    train_loss /= len(train_loader)
    # val_loss /= len(val_loader)

    logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={epoch_acc/cnt*100:.2f}%, val_loss={val_loss}, lr={scheduler.get_last_lr()[0]}')
    writer.add_scalars('loss',{'train':train_loss,'val':val_loss},epoch)
    cnt = 0

# finally evaluate model on the test set here
# ...

# save the trained network
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
ckpt_path = os.path.join(output_dir, f'trained-net-{cur_version}.pt')
torch.save(net.state_dict(), ckpt_path)
logger.info(f'Save ckpt in {ckpt_path}')
