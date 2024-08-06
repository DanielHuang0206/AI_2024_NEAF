import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt

## first part : Create data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # 增加一個維度！pytorch 中會需要多增加一個維度
y = x.pow(2) + 0.2 * torch.rand(x.size()) # add some noise

## second part : Create NN

class Net(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # To inherit things from Net, the standard process must be added

    def forward(self, x):
        
        return x 
    
net = Net(n_feature=1, n_hidden=10, n_output=1) 
print(net) # network architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion() # something about plotting

for t in range(200):
    

    if t % 5 == 0: 
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()