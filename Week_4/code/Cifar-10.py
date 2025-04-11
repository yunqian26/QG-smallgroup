import torch
import torchvision
from nltk import accuracy
from param import output
from torch import nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms. RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

test_data_set= torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((32,32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

train_data_load =DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True)
test_data_load =DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True)

train_data_set=len(train_data_set)
test_data_set=len(test_data_set)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络的主要部分
        self.main = nn.Sequential(
            # 第一个卷积层，输入通道数为3，输出通道数为32，卷积核大小为3，步长为1，填充为1
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            # 最大池化层，池化核大小为2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 批标准化层，输入通道数为32
            nn.BatchNorm2d(num_features=32),

            # 第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1，填充为1
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            # 最大池化层，池化核大小为2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 批标准化层，输入通道数为64
            nn.BatchNorm2d(num_features=64),

            # 第三个卷积层，输入通道数为64，输出通道数为128，卷积核大小为3，步长为1，填充为1
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            # 最大池化层，池化核大小为2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            # 批标准化层，输入通道数为128

        )

        self.fc=nn.Sequential(
        # 定义全连接层
            nn.Flatten(),
            # 展平操作

            nn.Linear(128*4*4, 1024), nn.ReLU(inplace=True),
            # 第一个全连接层，输入维度为128*4*4，输出维度为1024
            nn.Dropout(),
            # Dropout层，用于防止过拟合

            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            # 第二个全连接层，输入维度为1024，输出维度为256
            nn.Dropout(),
            # Dropout层，用于防止过拟合

            nn.Linear(256, 10),
            # 第三个全连接层，输入维度为256，输出维度为10
        )

    def forward(self, x):
    # 定义前向传播函数
        return self.fc(self.main(x))
        # 将输入数据传入网络的主要部分

mynet=MyNet()
mynet = mynet.to(device)
print(mynet)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate=1e-3
optim =torch.optim.Adam(mynet.parameters(), lr=learning_rate)

train_step =0
epochs =20

if __name__ == '__main__':
    for i in range(epochs):
        mynet.train()

        for j, (imgs,targets) in enumerate(train_data_load):

            imgs = imgs.to(device)
            targets = targets.to(device)

            output = mynet(imgs)
            loss =loss_fn(output, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_step +=1
            if train_step % 100 == 0:
                print('train_step:',train_step,'loss:',loss.item())

        mynet.eval()
        accuracy=0
        accuracy_total=0
        with torch.no_grad():
            for j, (imgs,targets) in enumerate(test_data_load):

                imgs = imgs.to(device)
                targets = targets.to(device)

                output = mynet(imgs)

                accuracy = (output.argmax(1) == targets).sum()
                accuracy_total +=accuracy

            print(f'第{i+1}轮训练结束，准确率为{accuracy_total / test_data_set}')
            torch.save(mynet, f'CIFAR_10_{i+1}_acc_{accuracy_total/test_data_set}.pth')






