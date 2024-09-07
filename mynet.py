import torch
import torch.nn as nn
class mynet(nn.Module):
    def __init__(self,feature_extractor,class_num):
        super(mynet, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc1 = nn.Linear(512, class_num)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x
    def incremental_learning(self, class_num):
        in_features = self.fc1.in_features
        out_features = self.fc1.out_features
        weight = self.fc1.weight.data
        bias = self.fc1.bias.data
        self.fc1 = nn.Linear(in_features, class_num)
        self.fc1.weight.data[:out_features] = weight
        self.fc1.bias.data[:out_features] = bias
        return self.fc1