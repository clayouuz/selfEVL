import torch
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import 
from utils.step_lr import StepLR

from utils.wide_res_net import WideResNet
from utils.smooth_cross_entropy import smooth_crossentropy
from mynet import network, toplayer
from utils.ResNet import resnet18_cbam
from utils.iCIFAR100 import iCIFAR100
from utils.log import Log
from utils.sam import SAM
from utils.bypass_bn import enable_running_stats, disable_running_stats

class selfEVL:
    def __init__(self,args,task_size,device,file_name):
        self.args = args
        self.feature_extractor = None
        self.classifier = None
        self.old_classifier = None
        self.feature_extractors = []
        self.numclass = args.init_nc
        self.task_size = task_size
        self.task_id = 0
        self.device = device
        self.file_name = file_name
        self.radius=None
        self.prototype=None
        self.class_label= None
        self.log = Log(log_each=10,batch_size=args.batch_size,flash=self.args.log_flash)
        
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
    def setup_data(self, shuffle):
        '''
        设置类的序号，shuffle为True时，打乱类的顺序
        将
        '''
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
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
        if current_task == 0:
            classes = [0, self.numclass]
            self.feature_extractor = network(resnet18_cbam(),self.numclass)
            self.classifier = toplayer(self.numclass,self.numclass)
            # self.classifier = toplayer(512,self.numclass)
        else:
            classes = [self.numclass - self.task_size, self.numclass]
            self.feature_extractor = network(resnet18_cbam(),self.numclass)
            self.classifier.Incremental_learning(self.task_size,self.numclass)
            # self.classifier.Incremental_learning(512,self.numclass)
        
        self.task_id = current_task
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
    def _train_feature(self):
        args=self.args
        log = self.log
        model = self.feature_extractor.to(self.device)
        model.train()
        
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        scheduler = StepLR(optimizer, args.lr, args.f_epochs)

        for epoch in range(args.f_epochs):
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
                for i, (_, inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
        log.flush()
        log.next_round()
    def _save_feature_extractor(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        path=path+'feature_{}_{}.pth'.format(self.task_size,self.numclass)
        # torch.save(self.feature_extractor.state_dict(), path)
        torch.save(self.feature_extractor.state_dict(), path)   
    def _train_classifier(self):
        model=self.classifier.to(self.device)
        model.train()
        args=self.args
        log=self.log
        test_up2now = self._get_test_dataloader([0, self.numclass])  
        optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr,weight_decay=2e-4)
        scheduler = StepLR(optimizer, args.lr, args.c_epochs)
        
        for epoch in range(self.args.c_epochs):
            model.train()
            log.train(len_dataset=len(self.train_loader))
            for i, (_, images, targets) in enumerate(self.train_loader):
                
                #optional: 4*rotation
                
                # images = torch.stack(
                #     [torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                # images = images.view(-1, 3, 32, 32)
                # targets = torch.stack([targets * 4 + k for k in range(4)],1).view(-1)
                
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                inputs = self._get_features(images)

                loss = self._loss(inputs, targets)
                loss.mean().backward()
                optimizer.step()

                with torch.no_grad():
                    outputs = model(inputs)
                    correct = torch.argmax(outputs.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    if epoch > 0:
                        scheduler(epoch)

            model.eval()
            log.eval(len_dataset=len(test_up2now))

            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(test_up2now):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    features=self._get_features(inputs)
                    predictions = model(features)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
        log.flush()
        log.next_round()                
    def _save_classifier(self):
        self.old_classifier = toplayer(self.numclass,self.numclass)
        self.old_classifier.load_state_dict(self.classifier.state_dict())
        self.old_classifier.to(self.device)
        self.protoSave(self.classifier, self.train_loader)

    def _loss(self, inputs, targets,smoothing=0.1):
        outputs = self.classifier(inputs)
        # get one-hot targets with smoothing
        n_class = outputs.size(1)
        one_hot = torch.full_like(outputs,fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=targets.unsqueeze(1).long(), value=1.0 - smoothing)
        targets_smooth = one_hot
        
        # cross entropy loss
        loss_cls = F.kl_div(F.log_softmax(outputs, dim=1), targets_smooth, reduction='none').sum(-1)
        loss = loss_cls
        
        if self._is_first_task():
            return loss
        
        # kd loss
        old_class = self.numclass-self.task_size
        old_output = self.old_classifier(inputs)
        new_output = outputs[:,:old_class]
        loss_kd = F.kl_div(F.log_softmax(new_output, dim=1), F.softmax(old_output, dim=1), reduction='none').sum(-1)
        
        loss+=loss_kd*self.args.kd_weight
        
        # prototype loss
        proto_aug = []
        proto_aug_label = []
        index = list(range(old_class))
        for _ in range(self.args.batch_size):
            np.random.shuffle(index)
            temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
            proto_aug.append(temp)
            proto_aug_label.append(4*self.class_label[index[0]])
        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
        soft_feat_aug = self.model.fc(proto_aug)
        loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label)
        
        loss+=loss_protoAug*self.args.protoAug_weight
        
        return loss
    
    def _get_features(self, inputs):# load feature extractor from state_dict
        output_list=[]
        for index,feature in enumerate(self.feature_extractors):
            model=network(resnet18_cbam(),self.args.init_nc+index*self.task_size)
            model.load_state_dict(feature)
            model.to(self.device)
            model.eval()
            output=model(inputs)
            output_list.append(output)
        for index,output in enumerate(output_list):
            if index==0:
                output=output[:,0:self.args.init_nc]
                temp=torch.clone(output)
            else:
                output=output[:,self.args.init_nc+(index-1)*self.task_size:self.args.init_nc+index*self.task_size]
                temp=torch.cat((temp,output),dim=1)
        return temp
    
    def _is_first_task(self):
        return self.numclass==self.args.init_nc
    
    def _get_feature_net(self):# load feature extractor from file
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        path=path+'feature_{}_{}.pth'.format(self.task_size,self.numclass)
        if os.path.exists(path):
            self.feature_extractor.load_state_dict(torch.load(path))
            print('load existed feature exatractor:',path)
            
        else:
            print('train feature extractor')
        return os.path.exists(path)
    
    def protoSave(self, model, loader):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if self._is_first_task():
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if self._is_first_task():
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            
    def train(self):
        if not self._get_feature_net():
            self._train_feature()
            self._save_feature_extractor()
        self.feature_extractors.append(self.feature_extractor.state_dict())
        
        print('train classifier')
        self._train_classifier()
        self._save_classifier()
 
        self.numclass+=self.task_size