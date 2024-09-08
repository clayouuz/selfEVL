import torch
import os
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import 
from utils.step_lr import StepLR

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from mynet import network
from utils.ResNet import resnet18_cbam
from utils.iCIFAR100 import iCIFAR100
from utils.log import Log
from utils.sam import SAM
from utils.bypass_bn import enable_running_stats, disable_running_stats

class selfEVL:
    def __init__(self,args,task_size,device,file_name):
        self.args = args
        self.feature_extractor = None
        # self.model = mynet(self.feature_extractor, 100)
        # self.model=network(100,resnet18_cbam())
        self.classifier = None
        self.feature_extractors = []
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.file_name = file_name
        self.log = Log(log_each=10,batch_size=args.batch_size)
        
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None
        

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
        # self.model.eval()  #设置为评估模式
        # self.feature_extractor = WideResNet(self.args.depth, self.args.width_factor, self.args.dropout, in_channels=3, labels=100).to(self.device)
        # self.feature_extractor = mynet(self.feature_extractor, self.numclass)
        self.feature_extractor = network(self.numclass,resnet18_cbam())
        
        
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            print(self.numclass)
            print(classes)
            # self.model.Incremental_learning(self.numclass)
        # self.model.train()  #设置为训练模式
        # self.model.to(self.device)
    
    def _train_feature(self):
        args=self.args
        log = self.log
        model = self.feature_extractor
        model.to(self.device)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        scheduler = StepLR(optimizer, args.lr, args.epochs)

        for epoch in range(args.epochs):
            model.train()
            log.train(len_dataset=len(self.train_loader))

            # for batch in self.train_loader:
            for i, (_, inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs).to(self.device)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                # loss = nn.CrossEntropyLoss()(predictions / self.args.label_smoothing,targets.long())
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                # output= model(inputs)
                # predictions = nn.Softmax(dim=1)(output)
                # nn.CrossEntropyLoss()(predictions / self.args.label_smoothing,targets.long()).mean().backward()
                optimizer.second_step(zero_grad=True)

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    # print(scheduler.get_last_lr())
                    if epoch > 0:
                        # scheduler.step()
                        scheduler(epoch)

            model.eval()
            log.eval(len_dataset=len(self.test_loader))

            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(self.train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
        log.flush()
        log.next_round()

        
        
    
    def _train_classifier(self):
        model=self.model
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr,weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=45, gamma=0.1)
        for epoch in range(self.args.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                
                images = torch.stack(
                    [torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                labels = torch.stack([labels * 4 + k for k in range(4)], # type: ignore
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
                with torch.no_grad():
                    correct = torch.argmax(outputs.data, 1) == labels
                    self.log(model, loss.cpu(), correct.cpu(), scheduler.lr())
            scheduler.step()
            
    def _loss(self, outputs, labels):
        # return F.cross_entropy(outputs, labels)
        return nn.CrossEntropyLoss()(outputs, labels.long())
    
    def train(self):

        self._train_feature()
        # self._train_classifier()

    def afterTrain(self):
        
        
        #save feature extractor
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        path=path+'{}:{}.pth'.format(self.task_size,self.numclass)
        torch.save(self.feature_extractor.state_dict(), path)
        #save classifier
        
        #save prototype
        
        self.numclass+=self.task_size
        pass
    
    def _test(self):
        
        # self.model.eval()
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