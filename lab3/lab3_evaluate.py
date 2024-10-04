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
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self, input_size, num_cls, hidden_layer_list, activate_func='relu'):
        super(Net, self).__init__()

        if activate_func == 'relu':
            self.activate = nn.ReLU()
        elif activate_func == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.1)
        elif activate_func == 'sigmoid':
            self.activate = nn.Sigmoid()
        elif activate_func == 'tanh':
            self.activate = nn.Tanh()

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

    
def main():
    resume_version = 'trained-net-20240507-001102-trainacc-57.58-valacc-51.77-testacc-50.87'
    resume_ckpt_path = f'output/{resume_version}.pt'

    # Resume the configurations
    config_file = f'output/{resume_version}.json'
    config = json.load(open(config_file, 'rb'))

    stateList = np.load('state_list.npy').tolist()
    num_cls = len(stateList)

    input_size = config['input_size']
    hidden_layer_list = config['hidden_layer_list']
    activate_func = config['activate_func']
    use_dynamic_features = config['use_dynamic_features']
    feature_type = config['feature_type']
    batch_size = 1024   # to speed up the inference

    # instantiate the network
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    net = Net(input_size=input_size, num_cls=num_cls, hidden_layer_list=hidden_layer_list, activate_func=activate_func)
    net = net.to(device)

    state_dict = torch.load(resume_ckpt_path)
    net.load_state_dict(state_dict)

    # prepare dataset
    prepared_data_test = np.load('prepared_test_data.npz')
    if use_dynamic_features:
        np_test_x = prepared_data_test[f'data_x_dynamic_{feature_type}']
        np_test_y = prepared_data_test['data_y']
    else:
        np_test_x = prepared_data_test[f'data_x_{feature_type}']
        np_test_y = prepared_data_test['data_y']
    test_x = torch.tensor(np_test_x)
    test_y = F.one_hot(torch.tensor(np_test_y, dtype=torch.long), num_classes=num_cls).float()
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # finally evaluate model on the test set here
    net.eval()
    test_cnt = 0
    logger.info('Testing...')
    all_outputs = np.zeros(len(test_dataset))
    all_labels = np.zeros(len(test_dataset))
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # move data from cpu to gpu (if available)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            all_outputs[test_cnt:test_cnt+len(outputs)] = torch.max(torch.softmax(outputs, dim=1), dim=1)[1].cpu().numpy()
            all_labels[test_cnt:test_cnt+len(outputs)] = torch.max(labels, dim=1)[1].cpu().numpy()
            test_cnt += len(inputs)
        
    # Evaluation: frame-by-frame at the state level
    correct_cnt = 0
    for idx in tqdm(range(len(all_outputs))):
        output = all_outputs[idx]
        label = all_labels[idx]
        if output == label:
            correct_cnt += 1
    test_acc = correct_cnt/len(all_outputs)
    logger.info(f'Frame-by-frame at the state level accuracy: {test_acc * 100:.2f}%')
    cm = confusion_matrix(all_labels, all_outputs)
    logger.info(f"State level confusion matrix:\n{cm}")
    
    output_dir = 'output'
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=stateList, yticklabels=stateList, cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    file_path = os.path.join(output_dir, 'confusion_matrix_state.png')
    plt.savefig(file_path)
    plt.close()

    # Evaluation: frame-by-frame at the phoneme level
    states = set([state.split('_')[0] for state in stateList])
    
    phoneme_dict = {state:[] for state in states}
    idx2phoneme = {}
    for state_idx, state in enumerate(stateList):
        phoneme_dict[state.split('_')[0]].append(state_idx)
        idx2phoneme[state_idx] = state.split('_')[0] 
    
    phoneme_outputs = [idx2phoneme[int(output)] for output in all_outputs]
    phoneme_labels = [idx2phoneme[label] for label in all_labels]
    
    correct_cnt = sum(1 for i, output in enumerate(phoneme_outputs) if output == phoneme_labels[i])
    test_acc = correct_cnt / len(phoneme_outputs)
    logger.info(f'Frame-by-frame at the phoneme level accuracy: {test_acc * 100:.2f}%')
    
    unique_phonemes = sorted(set(phoneme_labels))
    cm_phoneme = confusion_matrix(phoneme_labels, phoneme_outputs, labels=unique_phonemes)
    logger.info(f"Phoneme level confusion matrix:\n{cm_phoneme}")
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_phoneme, annot=False, fmt='d', cmap='Blues', xticklabels=unique_phonemes, yticklabels=unique_phonemes, cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    file_path = os.path.join(output_dir, 'confusion_matrix_phoneme.png')
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    main()