import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
import matplotlib.pyplot as plt
import model
import math
from torch.utils.data import DataLoader
from dataset import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def transform_scaler(path_list):
    scaler = MinMaxScaler()
    train_data = pd.read_csv(path_list[0])
    test_data = pd.read_csv(path_list[1])
    data = pd.concat([train_data, test_data], axis = 0)
    data = np.array(data)
    scaler.fit(data[:, 2:])
    return scaler

def train(model, device, train_dataloader, optimizer, epoch, loss_fn):
    model.train()
    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.float().to(device)
        output = model(data)
        output = output.squeeze(-1)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 500 == 0:
            print("Train Epoch:{}, iteration:{}, Loss:{}".format(epoch, idx, loss.item()))

def test(model, device, dataloader,loss_fn):
    model.eval()
    total_loss = 0.
    total_len = len(dataloader.dataset)
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.float().to(device)
            output = model(data)
            output = output.squeeze(-1)
            total_loss += loss_fn(output, target).item()
            
    total_loss = total_loss / total_len
    if dataloader == train_dataloader:
        print("Training loss is {}".format(total_loss)) 
    else:
        print("Test loss is {}".format(total_loss))
    return total_loss


if __name__ == "__main__":

    path_list = ['000001.XSHE_2014_to_2019_ti.csv', '000001.XSHE_2020_ti.csv']

    train_set = StockDataset(file_path = path_list[0], transform = transform_scaler(path_list))
    test_set = StockDataset(file_path = path_list[1], transform = transform_scaler(path_list))
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.MSELoss()  
    model = model.resnet32()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    test_total_loss = []
    train_total_loss = []

    for epoch in range(num_epochs):
        train(model, device, train_dataloader, optimizer, epoch, loss_fn)
        test_loss = test(model, device, test_dataloader,loss_fn)
        test_total_loss.append(test_loss)
        if test_loss == min(test_total_loss):
            torch.save(model.state_dict(),"best_cnn.pt")
        train_loss = test(model, device, train_dataloader,loss_fn)
        train_total_loss.append(train_loss)


    plt.plot(range(num_epochs),[test_total_loss[i] for i in range(num_epochs)])
    plt.plot(range(num_epochs),[train_total_loss[i] for i in range(num_epochs)])
    plt.legend(['test loss', 'training loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('train and test loss.jpg')



    
