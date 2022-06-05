import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict


metaFileName = 'cifar100/cifar-100-python/meta'
meta = unpickle(metaFileName)
fineLabelList = []
result = {}

for value in meta.get(b'fine_label_names'):
    fineLabelList.append(value.decode('utf-8'))
for item in fineLabelList:
    result[str(item)] = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
'''   
trainTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0,1), contrast=(0,1), saturation=(0,1), hue=0),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]) 
'''
testTransform1 = transforms.Compose([
    transforms.ToTensor(),
])
testTransform2 = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainDataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=trainTransform, download=True)
testDataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=testTransform1, download=True)

trainLoader = DataLoader(trainDataset, batch_size=200, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=200, shuffle=False)




def train():
    net.train()
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    for batch, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        net.optimizer.zero_grad()
        output = net(data)
        loss = net.lossFunc(output, target)
        loss.backward()
        net.optimizer.step()
        acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        sum += len(target)
        loss_sum += loss.item()
    writer.add_scalar('Cifar100_model_log/trainAccuracy', 100 * acc / sum, epoch + 1)
    writer.add_scalar('Cifar100_model_log/trainLoss', loss_sum / (batch + 1), epoch + 1)
    print('train accuracy: %.2f%%, loss: %.4f' % (100 * acc / sum, loss_sum / (batch + 1)))


def test():
    net.eval()
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    step = 0
    for batch, (data, target) in enumerate(testLoader):
        initData = data
        data = testTransform2(data)
        data, target = data.to(device), target.to(device)
        output = net(data)

        '''
        for i in range(len(output)):
           writer.add_image(fineLabelList[torch.argmax(output, dim=1)[i]], initData[i], step)
        step = step + 1
        '''

        loss = net.lossFunc(output, target)
        acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        sum += len(target)
        loss_sum += loss.item()
    writer.add_scalar('Cifar100_model_log/testAccuracy', 100 * acc / sum, epoch + 1)
    writer.add_scalar('Cifar100_model_log/trainLoss', loss_sum / (batch + 1), epoch + 1)
    print('test accuracy: %.2f%%, loss: %.4f' % (100 * acc / sum, loss_sum / (batch + 1)))





class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        self.downSample = nn.Sequential()
        if in_channels != out_channels:
            self.downSample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.network(x) + self.downSample(x)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock(3, 16)
        self.block2 = BasicBlock(16, 64, 2)
        self.block3 = BasicBlock(64, 64)
        self.block4 = BasicBlock(64, 128, 2)

        self.block5 = nn.Sequential()
        self.block6 = nn.Sequential()
        self.block7 = nn.Sequential()
        self.block8 = nn.Sequential()
        self.block9 = nn.Sequential()
        self.cov1 = nn.Sequential()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 2048),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 100)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.08)
        self.lossFunc = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        out = f.relu(self.block1(x))
        out = f.dropout(out, 0.1)
        out = f.relu(self.block2(out))
        out = f.dropout(out, 0.1)
        out = torch.sigmoid(self.block3(out))
        out = f.dropout(out, 0.1)
        out = torch.relu(self.block4(out))
        out = f.dropout(out, 0.1)
        out = f.relu(self.block5(out))
        out = f.dropout(out, 0.1)
        out = f.relu(self.block6(out))
        out = f.dropout(out, 0.1)
        out = f.relu(self.block7(out))
        out = f.relu(self.block8(out))
        out = f.relu(self.block9(out))
        out = f.relu(self.cov1(out))
        out = f.dropout(out, 0.1)
        out = self.linear(out)
        return out


'''
第一次迁移学习:
冻结block1~4
增加
block5：BasicBlock(128, 128) 
block6：BasicBlock(128, 512, 2)
重设全连接层
正确率浮动变化不大

第二次迁移学习：
第一次的基础上
补充cov1单层卷积2*2*512
（如果增加至三层则会下降严重）
正确率提升至58%

第三次迁移学习
原版基础上重设全连接层单层
目的为了训练卷积层
'''

net = Net()
writer = SummaryWriter(log_dir='Cifar100_model_log')

if __name__ == '__main__':
    try:
        net = torch.load("Cifar100_model.pkl")
        print("Start retrain :")
        for parm in net.parameters():
            parm.requires_grad = True
        net.optimizer = torch.optim.SGD(net.parameters(), lr=0.08)
        net = net.to(device)
        for epoch in range(60):
            print('epoch: %d' % (epoch + 1))
            train()
            test()
        torch.save(net, 'Cifar100_model.pkl')
        writer.close()

    except:
        print("Start train :")
        with SummaryWriter(log_dir='Cifar100_model_log') as w:
            w.add_graph(net, torch.from_numpy(np.reshape(trainDataset.data[0], (-1, 3, 32, 32))).to(torch.float32))
        net = net.to(device)
        for epoch in range(50):
            print('epoch: %d' % (epoch + 1))
            train()
            test()
        torch.save(net, 'Cifar100_model.pkl')
        writer.close()

