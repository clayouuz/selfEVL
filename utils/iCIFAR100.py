from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False,
                 testmode=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.testmode = testmode

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        # if self.testmode:
        #     self.TestData = self.TestData[:100]
        #     self.TestLabels = self.TestLabels[:100]
        #     print("now in test mode, only use 100 samples for testing")
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))

    def getTrainData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        # if self.testmode:
        #     #for fast test, only use 1000 samples
        #     self.TrainData = self.TrainData[:1000]
        #     self.TrainLabels = self.TrainLabels[:1000]
        #     print("now in test mode, only use 1000 samples for training")
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) != 0:
            return self.getTrainItem(index)
        elif len(self.TestData) != 0:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) != 0:
            return len(self.TrainData)
        elif len(self.TestData) != 0:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

