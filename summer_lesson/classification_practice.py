import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#創建數據
n_data = torch.ones(100, 2)#創建一個100x2的全1張量，作為基礎數據
x0 = torch.normal(2*n_data,1)#創建一個均值為2標準差為一的正態分布數據，大小為100x2
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)#創建一個長度為100的標籤張量，標籤為1
x = torch.cat((x0,x1),0).type(torch.FloatTensor)#將x0和x1沿著第0維度拼接在一起，並轉換為浮點張量
y = torch.cat((y0,y1),).type(torch.LongTensor)#將y0和y1拼接在一起，並轉換為長整型張量
#第二部分:建立神經網路
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net,self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature,10)
        self.hidden_2 = torch.nn.Linear(10,8)
        self.out = torch.nn.Linear(8,n_output)
    
    def forward(self,x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_output=2)  #建立神經網路，輸入特徵向量為2，輸出類別向量為2
print(net)

optimazer = torch.optim.SGD(net.parameters(), lr = 0.02)    #使用SGD為優化器，學習率為0.02
loss_func = torch.nn.CrossEntropyLoss() #損失函數選用交叉商損失

plt.ion()
epochs =  100
for t in range(epochs):
    out = net(x)
    loss = loss_func(out,y)

    optimazer.zero_grad()
    loss.backward()
    optimazer.step()

    if t%5 == 0:
        plt.cla()   #清除當前圖像
        prediction = torch.max(out,1)[1]
        pred_y = y.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y,s = 100, lw=0, cmap = 'RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum())/float(target_y.size)
        plt.text(1.5, -4,'Accuracy=%2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
"""
import torch  # 導入PyTorch庫，用於深度學習模型的構建和訓練
import torch.nn.functional as F  # 導入PyTorch中的函數模塊，包括激活函數和損失函數
import matplotlib.pyplot as plt  # 導入Matplotlib的pyplot模塊，用於數據可視化

## 第一部分：創建數據
x0 = torch.normal(2 * n_data, 1)   # 創建一個均值為2，標準差為1的正態分布數據，大小為100x2
y0 = torch.zeros(100)  # 創建一個長度為100的標籤張量，標籤為0
x1 = torch.normal(-2 * n_data, 1)  # 創建一個均值為-2，標準差為1的正態分布數據，大小為100x2
y1 = torch.ones(100)  # 創建一個長度為100的標籤張量，標籤為1
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 將x0和x1沿著第0個維度拼接在一起，並轉換為浮點張量
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # 將y0和y1拼接在一起，並轉換為長整型張量

## 第二部分：創建神經網絡

class Net(torch.nn.Module):  # 定義一個神經網絡類，繼承自torch.nn.Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()  # 調用父類的初始化函數
        self.hidden_1 = torch.nn.Linear(n_feature, 10)  # 定義第一個隱藏層，全連接層，有10個神經元
        self.hidden_2 = torch.nn.Linear(10, 8)  # 定義第二個隱藏層，全連接層，有8個神經元
        self.out = torch.nn.Linear(8, n_output)  # 定義輸出層，全連接層，輸出有2個神經元

    def forward(self, x):
        x = F.relu(self.hidden_1(x))  # 對第一個隱藏層的輸出應用ReLU激活函數
        x = F.relu(self.hidden_2(x))  # 對第二個隱藏層的輸出應用ReLU激活函數
        x = self.out(x)  # 輸出層不應用激活函數，直接輸出結果
        return x  # 返回最終的輸出

net = Net(n_feature=2, n_output=2)  # 創建神經網絡實例，輸入特徵數量為2，輸出類別數量為2
print(net)  # 打印網絡結構

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 使用隨機梯度下降法（SGD）作為優化器，學習率為0.02
loss_func = torch.nn.CrossEntropyLoss()  # 使用交叉熵損失函數，適用於分類問題，已經隱含了softmax激活函數

plt.ion()  # 打開Matplotlib的交互模式，允許動態更新圖表

for t in range(100):  # 訓練循環，進行100次迭代
    out = net(x)  # 將輸入數據x傳入網絡，得到輸出
    loss = loss_func(out, y)  # 計算輸出out與真實標籤y之間的損失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向傳播，計算梯度
    optimizer.step()  # 更新網絡參數

    if t % 5 == 0:  # 每5次迭代打印一次結果並更新圖像
        plt.cla()  # 清除當前圖像
        prediction = torch.max(out, 1)[1]  # 獲取每個數據點的預測標籤
        pred_y = prediction.data.numpy()  # 將預測張量轉換為NumPy陣列
        target_y = y.data.numpy()  # 將真實標籤張量轉換為NumPy陣列
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')  # 繪製散點圖，顯示數據點
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)  # 計算準確率
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})  # 在圖上顯示準確率
        plt.pause(0.1)  # 暫停0.1秒，更新圖像

plt.ioff()  # 關閉交互模式
plt.show()  # 顯示最終圖像
"""