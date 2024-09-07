import torch

from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ResNet import resnet18_cbam
import numpy as np
from mynet import mynet
from utils.iCIFAR100 import iCIFAR100
import numpy as np

class selfEVL:
    def __init__(self,args,task_size,device,file_name):
        self.args = args
        self.model = mynet(None, 100)
        self.feature_extractor = None
        self.feature_extractors = []
        self.numclass = 100
        self.task_size = task_size
        self.device = device
        self.file_name = file_name
        
        self.train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),  #随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转
            transforms.ColorJitter(brightness=0.24705882352941178),  #随机改变图像的亮度
            transforms.ToTensor(),  #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761))  #标准化，参数来自对数据集的计算，分别是均值和标准差，rgb元组
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        self.train_dataset = iCIFAR100('./dataset',
                                       transform=self.train_transform,
                                       download=True,
                                       testmode=args.testmode)
        self.test_dataset = iCIFAR100('./dataset',
                                      test_transform=self.test_transform,
                                      train=False,
                                      download=True,
                                      testmode=args.testmode)
        

    def map_new_class_index(self, y, order):
        '''
        元素按照order的顺序进行重新排列
        y: list of class labels
        order: list of class labels in new order
        return: list of class labels in new order
        '''
        return np.array(list(map(lambda x: order.index(x), y)))
    def setup_data(self, shuffle, seed):
        '''
        设置类的序号，shuffle为True时，打乱类的顺序
        将
        '''
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100 * '#')
        print(self.class_order)
        self.train_dataset.targets = self.map_new_class_index( # type: ignore
            train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index( # type: ignore
            test_targets, self.class_order)        
    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def beforeTrain(self, current_task):
        self.model.eval()  #设置为评估模式

        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass - self.args.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(
            classes)
        if current_task > 0:
            self.model.Incremental_learning(4 * self.numclass)
        self.model.train()  #设置为训练模式
        self.model.to(self.device)
    
    def _train_feature(self):
        model = resnet18_cbam()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=45, gamma=0.1)
        for epoch in range(self.args.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                
                images = torch.stack(
                    [torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], # type: ignore
                                     1).view(-1)
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %
                          (epoch + 1, self.args.epochs, i + 1, len(self.train_loader), loss.item()))
            scheduler.step()
    
    def _train_classifier(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr,weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=45, gamma=0.1)
        for epoch in range(self.args.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                
                images = torch.stack(
                    [torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], # type: ignore
                                     1).view(-1)
                
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self._loss(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %
                          (epoch + 1, self.args.epochs, i + 1, len(self.train_loader), loss.item()))
            scheduler.step()
            
    def _loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
    
    def train(self):

        self._train_feature()
        self._train_classifier()

    def afterTrain(self):
        #save feature extractor
        
        #save classifier
        
        #save prototype
        
        pass
    
    def _test(self):
        
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            outputs = outputs[:, ::4]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))