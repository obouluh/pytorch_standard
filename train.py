# 倒入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from dataset1 import MyDataset
from model1 import MyModel
from torch.utils.data import DataLoader

# 超参数
epochs = 10
lr = 0.01
momentum=0.9
batch_size = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset = MyDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MyDataset()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model = MyModel()
model.to(device)# 移动到gpu0

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr,momentum=momentum)

for epoch in range(epochs):
    model.train()# 设置为训练模型，可以启动drpout等功能
    train_loss = 0
    train_acc = 0 
    for i, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # 优化器中的梯度清零

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step() # 根据梯度更新参数

        train_loss += loss.item() * inputs.size(0) # 累加批次的损失

        _, preds = torch.max(outputs, 1) #根据输出得到预测类别

        train_acc+= torch.sum(preds == labels).item()# 累加批次的正确预测数

    train_loss = train_loss / len(train_dataset)

    train_acc = train_acc / len(train_dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}') # 打印训练结果

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():# 关闭梯度计算、节省内存加快计算
        for datas in test_loader:
            inputs,labels = datas
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0) # 累加批次的损失

            _, preds = torch.max(outputs, 1) #根据输出得到预测类别

            test_acc+= torch.sum(preds == labels).item()# 累加批次的正确预测数

        test_loss = test_loss / len(test_dataset)

        test_acc = test_acc / len(test_dataset)

        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}') # 打印训练结果

    # 保存模型，包括结构和参数
    # path1 = ''
    # torch.save(model, path1)
    # model = torch.load(path1)

    # 保存模型的状态字典，文件小
    # path2 = ''
    # torch.save(model.state_dict(), path2)
    # model = MyModel()
    # model.load_state_dict(torch.load(path2))


import matplotlib.pyplot as plt
plt.plot(train_loss, label='Train Loss') # 绘制训练集损失曲线
plt.plot(test_loss, label='Valid Loss') # 绘制验证集损失曲线
plt.xlabel('Iterations') # 设置x轴标签
plt.ylabel('Loss') # 设置y轴标签
plt.legend() # 显示图例
plt.show() # 显示图像
 
plt.plot(train_acc, label='Train Acc') # 绘制训练集准确率曲线
plt.plot(test_acc, label='Valid Acc') # 绘制验证集准确率曲线
plt.xlabel('Iterations') # 设置x轴标签
plt.ylabel('Accuracy') # 设置y轴标签
plt.legend() # 显示图例
plt.show() # 显示图像
