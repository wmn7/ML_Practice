import time
import csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn import preprocessing
import copy

COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate"] # 共41个维度

# ---------------
# Visual The Loss
# ---------------
def draw_loss_acc(train_list,validation_list,mode='Loss'):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 1, 1)
    # 设置间隔
    data_len = len(train_list)
    x_ticks = np.arange(1,data_len+1)
    plt.xticks(x_ticks)
    if mode == 'Loss':
        plt.plot(x_ticks,train_list,label='Train Loss')
        plt.plot(x_ticks,validation_list,label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Epoch_loss.jpg')
    elif mode == 'Accuracy':
        plt.plot(x_ticks,train_list,label='Train Accuracy')
        plt.plot(x_ticks,validation_list,label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('Epoch_Accuracy.jpg')

# ----------------
# Train the model
# ----------------
def train_model(model, criterion, optimizer, dataloaders, train_datalengths, scheduler=None, num_epochs=2):
    """传入的参数分别是:
    1. model:定义的模型结构
    2. criterion:损失函数
    3. optimizer:优化器
    4. dataloaders:training dataset
    5. train_datalengths:train set和validation set的大小, 为了计算准确率
    6. scheduler:lr的更新策略
    7. num_epochs:训练的epochs
    """
    since = time.time()
    # 保存最好一次的模型参数和最好的准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = [] # 记录每一个epoch后的train的loss
    train_acc = []
    validation_loss = []
    validation_acc = []
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 # 这个是一个epoch积累一次
            running_corrects = 0 # 这个是一个epoch积累一次

            # Iterate over data.
            total_step = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # inputs = inputs.reshape(-1, 28*28).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 使用output(概率)得到预测
                    
                    loss = criterion(outputs, labels) # 使用output计算误差

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if (i+1)%100==0:
                    # 这里相当于是i*batch_size的样本个数打印一次, i*100
                    iteration_loss = loss.item()/inputs.size(0)
                    iteration_acc = 100*torch.sum(preds == labels.data).item() / len(preds)
                    print ('Mode {}, Epoch [{}/{}], Step [{}/{}], Accuracy: {}, Loss: {:.4f}'.format(phase, epoch+1, num_epochs, i+1, total_step, iteration_acc, iteration_loss))
            
            epoch_loss = running_loss / train_datalengths[phase]
            epoch_acc = running_corrects.double() / train_datalengths[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                validation_loss.append(epoch_loss)
                validation_acc.append(epoch_acc)
            print('*'*10)
            print('Mode [{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('*'*10)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('*'*10)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('*'*10)
    # load best model weights
    final_model = copy.deepcopy(model) # 最后得到的model
    model.load_state_dict(best_model_wts) # 在验证集上最好的model
    draw_loss_acc(train_list=train_loss,validation_list=validation_loss,mode='Loss') # 绘制Loss图像
    draw_loss_acc(train_list=train_acc,validation_list=validation_acc,mode='Accuracy') # 绘制准确率图像
    return (model,final_model)

if __name__ == "__main__":
    # --------------------
    # Device configuration
    # --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------
    # Hyper-parameters 
    # ----------------
    input_size = 122
    hidden_size = 100
    num_classes = 2
    n_layers = 5
    num_epochs = 10
    batch_size = 100
    validation_split = 0.05 # 每次训练集中选出10%作为
    learning_rate = 0.001

    # -------------
    # Train dataset 
    # -------------
    filepath = './ProcessData/Xfull/Yfull/One-hot_Two-classification/'
    #X_train_path = filepath+'X_train_my.csv'
    #X_test_path = filepath+'X_test_my.csv'
    #Y_train_path = filepath+'Y_train_my.csv'
    #Y_test_path = filepath+'Y_test_my.csv'

    X_test_path = filepath+'X_train_my.csv'
    X_train_path = filepath+'X_test_my.csv'
    Y_test_path = filepath+'Y_train_my.csv'
    Y_train_path = filepath+'Y_test_my.csv'

    X_train = pd.read_csv(X_train_path, index_col=False)
    X_test = pd.read_csv(X_test_path, index_col=False)
    Y_train = pd.read_csv(Y_train_path, index_col=False)
    Y_test = pd.read_csv(Y_test_path, index_col=False)

    # 数据做标准化
    X_train = preprocessing.scale(X_train.astype(float))
    X_test = preprocessing.scale(X_test.astype(float))

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train.values.squeeze()).long()
    Y_test = torch.from_numpy(Y_test.values.squeeze()).long()

    print("X_train.size:{}".format(X_train.size()))
    # -----------
    # DataLoader
    # -----------
    train_dataset = Data.TensorDataset(X_train, Y_train) # 合并训练数据和目标数据
    test_dataset = Data.TensorDataset(X_test, Y_test) # 合并训练数据和目标数据
    test_len = len(test_dataset)
    test_dataset = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2           # set multi-work num read data
    )

    # ------------------
    # 下面切分validation
    # ------------------
    dataset_len = len(train_dataset)
    indices = list(range(dataset_len))
    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len)) # validation的长度
    validation_idx = np.random.choice(indices, size=val_len, replace=False) # validatiuon的index
    train_idx = list(set(indices) - set(validation_idx)) # train的index
    ## Defining the samplers for each phase based on the random indices:
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                sampler=train_sampler,
                                                batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=validation_sampler,
                                                batch_size=batch_size)
    train_dataloaders = {"train": train_loader, "val": validation_loader} # 使用字典的方式进行保存
    train_datalengths = {"train": len(train_idx), "val": val_len} # 保存train和validation的长度


    # -------------------------------
    # Fully connected neural network
    # -------------------------------
    """
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, n_layers):
            super(NeuralNet, self).__init__()
            self.LGLayer = nn.Linear(input_size, num_classes) 
        
        def forward(self, x):
            out = self.LGLayer(x)
            return out
    """
    
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, n_layers):
            super(NeuralNet, self).__init__()
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size,momentum=0.5))
                layers.append(nn.Dropout(0.9))
                layers.append(nn.ReLU())
            self.inLayer = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU()
            self.hiddenLayer = nn.Sequential(*layers)
            self.outLayer = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out = self.inLayer(x)
            out = self.relu(out)
            out = self.hiddenLayer(out)
            out = self.outLayer(out)
            return out

    # 模型初始化
    model = NeuralNet(input_size, hidden_size, num_classes, n_layers).to(device)

    # 打印模型结构
    print(model)

    # -------------------
    # Loss and optimizer
    # ------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -------------
    # 进行模型的训练
    # -------------
    (best_model,final_model) = train_model(model=model,criterion=criterion,optimizer=optimizer,dataloaders=train_dataloaders,train_datalengths=train_datalengths,num_epochs=num_epochs)

    # --------------------------------
    # Test the model(在测试集上进行测试)
    # --------------------------------
    print('在测试集上进行测试.')
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model_list = [best_model,final_model]
    name_list = ['Best_model','Final_model']
    for name,model in zip(name_list,model_list):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in test_dataset:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the {} network on the {} test Data: {:.4f} %'.format(name, test_len, 100 * correct / total))