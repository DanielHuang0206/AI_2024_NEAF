import torch  # 導入PyTorch庫

## 第一部分：創建張量
x = torch.randn(3, 3, requires_grad=True)  # 創建一個3x3的隨機張量，並設置requires_grad=True，以便跟蹤其操作並計算梯度
print("Initual Tensor:\n", x)  # 輸出初始化的張量

## 第二部分：計算梯度
y = x + 2  # 對x進行加法操作，得到y，這也是一個張量並且requires_grad=True
z = y * y * 3  # 對y進行乘法操作，得到z，這也是一個張量並且requires_grad=True

out = z.mean()  # 將z的所有元素取平均，得到一個標量out
print("\nOutput:\n", out)  # 輸出計算結果out

out.backward()  # 計算out相對於x的梯度，將梯度存儲在x.grad中
print("\nGradiants:\n", x.grad)  # 輸出x的梯度

# 使用torch.no_grad()暫時禁用梯度計算
with torch.no_grad():
    y = x + 2  # 再次對x進行加法操作，此時不會跟蹤梯度
    print("\nTensor with no grad:\n", y)  # 輸出無梯度的張量
    # y.backward()  # 此行代碼被註釋掉，因為在不計算梯度的上下文中執行backward()會導致錯誤