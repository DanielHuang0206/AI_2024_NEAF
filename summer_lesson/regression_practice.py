import torch 
# 導入PyTorch庫，這是一個流行的深度學習框架。

import torch.nn.functional as F 
# 導入PyTorch中的功能模塊，包含許多常用的函數，例如激活函數、損失函數等。

import matplotlib.pyplot as plt
# 導入Matplotlib庫的pyplot模塊，用於數據可視化。

## 第一部分：創建數據
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
# 使用torch.linspace(-1, 1, 100)生成一個從-1到1的等間距張量，包含100個點，然後使用torch.unsqueeze增加一個維度。

y = x.pow(2) + 0.2 * torch.rand(x.size()) 
# 計算x的平方，並加上一些隨機噪聲，生成目標數據y。

## 第二部分：創建神經網絡
# 定義一個神經網絡類別，繼承自torch.nn.Module
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        # 調用父類的構造函數，初始化Net類的實例。

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 定義隱藏層，全連接層，從輸入層到隱藏層。

        self.predict = torch.nn.Linear(n_hidden, n_output)
        # 定義輸出層，全連接層，從隱藏層到輸出層。

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # 使用ReLU激活函數處理隱藏層的輸出。

        x = self.predict(x)
        # 將隱藏層的輸出傳遞到輸出層。

        return x 
        # 返回網絡的最終輸出。
    
net = Net(n_feature=1, n_hidden=10, n_output=1) 
# 創建Net類的實例，定義一個神經網絡，包含1個輸入特徵、10個隱藏單元和1個輸出單元。

print(net) 
# 打印網絡的結構。

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# 使用隨機梯度下降法作為優化器，設置學習率為0.2。

loss_func = torch.nn.MSELoss()
# 使用均方誤差作為損失函數。

plt.ion() 
# 打開交互模式，允許動態更新圖表。

# 訓練過程
for t in range(200):
    prediction = net(x)
    # 使用網絡對輸入數據x進行預測。

    loss = loss_func(prediction, y)
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

        plt.scatter(x.data.numpy(), y.data.numpy())
        # 繪製原始數據點。

        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # 繪製網絡的預測結果。

        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        # 在圖表上顯示當前的損失值。

        plt.pause(0.1)
        # 暫停0.1秒，更新圖表。

plt.ioff()
# 關閉交互模式。

plt.show()
# 顯示最終的圖表。
