import torch 
# 導入PyTorch庫，這是一個流行的深度學習框架。

import torch.nn.functional as F
# 導入PyTorch中的功能模塊，包含許多常用的函數，例如激活函數、損失函數等。

import matplotlib.pyplot as plt
# 導入Matplotlib庫的pyplot模塊，用於數據可視化。

## 第一部分：創建數據
n_data = torch.ones(100, 2)
# 創建一個形狀為(100, 2)的張量，每個元素都是1。

x0 = torch.normal(2 * n_data, 1)   # torch.Size([100, 2])
# 創建一個形狀為(100, 2)的正態分佈張量，均值為2，標準差為1。

y0 = torch.zeros(100)              # label = 0
# 創建一個長度為100的張量，所有元素都是0，作為類別標籤。

x1 = torch.normal(-2 * n_data, 1)  # torch.Size([100, 2])
# 創建一個形狀為(100, 2)的正態分佈張量，均值為-2，標準差為1。

y1 = torch.ones(100)               # label = 1 
# 創建一個長度為100的張量，所有元素都是1，作為類別標籤。

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # torch.Size([200, 2])
# 將x0和x1在第0維上進行拼接，形成一個形狀為(200, 2)的張量，並轉換為FloatTensor類型。

y = torch.cat((y0, y1), ).type(torch.LongTensor)    # torch.Size([200])
# 將y0和y1進行拼接，形成一個長度為200的張量，並轉換為LongTensor類型。

## 第二部分：創建神經網絡
class Net(torch.nn.Module): # 定義一個繼承自torch.nn.Module的神經網絡類
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__() 
        # 調用父類的構造函數，初始化Net類的實例。

        self.hidden_1 = torch.nn.Linear(n_feature, 10)
        # 定義第一個隱藏層，全連接層，從輸入層到隱藏層，有10個隱藏單元。

        self.hidden_2 = torch.nn.Linear(10, 8)
        # 定義第二個隱藏層，全連接層，從第一個隱藏層到第二個隱藏層，有8個隱藏單元。

        self.out = torch.nn.Linear(8, n_output)
        # 定義輸出層，全連接層，從隱藏層到輸出層，有2個輸出單元。

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        # 使用ReLU激活函數處理第一個隱藏層的輸出。

        x = F.relu(self.hidden_2(x))
        # 使用ReLU激活函數處理第二個隱藏層的輸出。

        x = self.out(x)
        # 將隱藏層的輸出傳遞到輸出層。

        return x 
        # 返回網絡的最終輸出。

net = Net(n_feature=2, n_output=2) 
# 創建Net類的實例，定義一個神經網絡，包含2個輸入特徵和2個輸出單元。

print(net) 
# 打印網絡的結構。

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 使用隨機梯度下降法作為優化器，設置學習率為0.02。

loss_func = torch.nn.CrossEntropyLoss() 
# 使用交叉熵作為損失函數，適用於分類問題。

plt.ion()
# 打開交互模式，允許動態更新圖表。

# 訓練過程
for t in range(100):
    out = net(x)
    # 使用網絡對輸入數據x進行預測。

    loss = loss_func(out, y)
    # 計算預測值和真實值之間的損失。

    optimizer.zero_grad()
    # 清除之前的梯度。

    loss.backward()
    # 反向傳播，計算梯度。

    optimizer.step()
    # 更新網絡參數。

    if t % 5 == 0:
        plt.cla()
        # 清除當前圖表。

        prediction = torch.max(out, 1)[1]    
        # 取得預測結果中每行最大值的索引。

        pred_y = prediction.data.numpy()     
        # 將張量轉換為NumPy數組。

        target_y = y.data.numpy()            
        # 將張量轉換為NumPy數組。

        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        # 繪製數據點，顏色根據預測結果設置。

        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)   
        # 計算預測的準確率。

        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        # 在圖表上顯示當前的準確率。

        plt.pause(0.1)
        # 暫停0.1秒，更新圖表。

plt.ioff()
# 關閉交互模式。

plt.show()
# 顯示最終的圖表。
