import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from os import walk

# 檢查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load data
train_x = []
train_x_std = []
train_y = []  # 有沒有震顫0或1
folder_name = ['Yes', 'No']

# 把資料讀進來
for folder in folder_name:
    path = 'C:/Users/user/Desktop/0730fork/AI_2024_NEAF/Data/' + str(folder) + '/'
    for root, dirs, files in walk(path):
        for f in files:
            filename = path + f
            print(filename)
            
            acc = scipy.io.loadmat(filename)
            acc = acc['tsDS'][:, 1].tolist()[0:7500]
            train_x.append(acc)
            train_x_std.append(np.std(acc))  # 標準差

            if folder == 'Yes':    
                train_y.append(1)
            elif folder == 'No':
                train_y.append(0)

# 資料以np.array的方式存入
train_x = np.array(train_x_std)
train_y = np.array(train_y)

# 最大變成1最小變0
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x.reshape(-1, 1))

# 將資料轉換為Torch張量並移動到GPU
train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.long).to(device)

# 定義神經網絡模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 第一層，全連接層
        self.fc2 = nn.Linear(64, 32) # 第二層，全連接層
        self.fc3 = nn.Linear(32, 2)  # 第三層，全連接層，輸出2個類別
        self.relu = nn.ReLU()        # 激活函數
        self.dropout = nn.Dropout(0.5)  # 添加Dropout以防止過擬合

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 初始化模型，並移動到GPU
model = Net().to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 交叉驗證 (Leave-One-Out)
loo = LeaveOneOut()
y_pred = []
y_true = []

i = 0
for train_idx, test_idx in loo.split(train_x):
    # 獲取訓練和測試數據
    x_train, x_test = train_x[train_idx], train_x[test_idx]
    y_train, y_test = train_y[train_idx], train_y[test_idx]

    # 創建數據集和加載器
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 訓練模式
    model.train()
    epoch_times = 500
    i = i+1
    print(i)
    for epoch in range(epoch_times):  # 可以增加epoch次數以獲得更好的結果
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            # 清空梯度
            optimizer.zero_grad()
            # 前向傳播
            output = model(batch_x)
            # 計算損失
            loss = criterion(output, batch_y)
            # 反向傳播
            loss.backward()
            # 更新參數
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epoch_times}, Loss: {running_loss/len(train_loader):.4f}")

    # 測試階段
    model.eval()
    with torch.no_grad():
        output = model(x_test)
        pred = torch.argmax(output, dim=1).item()
        y_pred.append(pred)
        y_true.append(y_test.item())

# 計算混淆矩陣
cf_m = confusion_matrix(y_true, y_pred)
print('confusion_matrix: \n', cf_m)

tn, fp, fn, tp = cf_m.ravel()
accuracy = (tn + tp) / (tn + tp + fn + fp)
print('Accuracy:', accuracy)
