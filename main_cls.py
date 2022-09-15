#%%
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pandas as pd
import torch
from torch import nn
import datetime
import time
import os
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DAYS_FOR_TRAIN = 10


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.bn = nn.BatchNorm1d(hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集
        
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y= [], []
    for i in range(len(data)-days_for_train-7):
        _x = data[i:(i+days_for_train)]
        dataset_x.append(_x)
        end_day = data[i+days_for_train+7] 
        start_data = data[i+days_for_train-1]
        dataset_y.append(end_day > (start_data*1.05))
    return (np.array(dataset_x), np.array(dataset_y))
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
def transDate(df):
    d,m,y = df.split('-')
    m = month.index(m)+1
    if m<10:
        m = '0' + str(m)
    else:
        m = str(m)
    d = int(d)
    if d<10:
        d = '0' + str(d)
    else:
        d = str(d)
    return y + '-' + m + '-' + d
#%%
data_close = pd.read_csv('apple_share_price.csv')['Close'] #读取文件
# data_close['Date'] = pd['Date'].apply(transDate)
#df_sh = ts.get_k_data('sh', start='2019-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))
#print(df_sh.shape)
data_close = data_close.astype('float64').values  # 转换数据类型

date = pd.read_csv('apple_share_price.csv')['Date']
date = date.values.tolist()
date.reverse()
close = data_close.tolist()
close.reverse()
data_close = np.array(close)
# %%
max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.33, random_state=42)
train_x = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
train_y = torch.tensor(y_train, dtype=torch.int64)
# %%
model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=3)
epochs = 100
criterier = torch.nn.BCELoss()

#%%
with tqdm(total=epochs) as t:
    for i in range(epochs):
        out = model(train_x).squeeze()
        loss = criterier(torch.sigmoid(out), train_y.to(torch.float32))
        loss.backward()
        with torch.no_grad():
            prd = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)).squeeze()
            y_pred = (prd>0.5).numpy()
            acc = np.mean(y_pred == y_test)
            t.set_postfix(loss = loss.item(), acc = acc)
        t.update(1)
        
#%%
prd = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)).squeeze()
y_pred = (prd>0.5).numpy()
from sklearn.metrics import classification_report
print("classification_report of LSTM model")
print(classification_report(y_test, y_pred, digits=4, target_names=['0', '1']))
# %%
from sklearn import neighbors
knc = neighbors.KNeighborsClassifier()
knc.fit(X_train,y_train)
prd = knc.predict(X_test)
print("classification_report of KNN model")
print(classification_report(y_test, prd, digits=4, target_names=['0', '1']))

#%%
from sklearn.svm import SVC
svm = SVC(kernel='rbf',random_state=1,gamma=100.0,C=1.0,verbose=1)   
svm.fit(X_train,y_train)
prd = svm.predict(X_test)
print("classification_report of SVM model")
print(classification_report(y_test, prd, digits=4, target_names=['0', '1']))