import os
from PIL import Image
import torchvision
from torch import nn
import torch.serialization

class MyNet(nn.Module):
    def __init__(self):
        # 初始化函数
        super().__init__()
        # 调用父类的初始化函数
        self.main = nn.Sequential(
            # 定义卷积层
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            # 定义最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 定义批归一化层
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),

        )

        self.fc=nn.Sequential(
            # 定义全连接层
            nn.Flatten(),

            nn.Linear(128*4*4, 1024), nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(self.main(x))

target_index={0:'飞机',1:'汽车',2:'鸟',3:'猫',4:'鹿',5:'狗',6:'青蛙',7:'马',8:'船',9:'卡车'}

root_dir='test_CIFAR_10'
obj_dir ='test.png'
img_dir=os.path.join(root_dir,obj_dir)
img =Image.open(img_dir)


tran_poss =torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.ToTensor()
])

torch.serialization.add_safe_globals([MyNet])
torch.serialization.safe_globals([MyNet])
mynet =torch.load('CIFAR_10_20_acc_0.818399965763092.pth',weights_only=False,map_location='cpu')
mynet.eval()
with torch.no_grad():
    img =tran_poss(img)
    img =torch.reshape(img,(1,3,32,32))

output=mynet(img)
print(target_index[output.argmax(1).item()])
