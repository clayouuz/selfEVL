import torch.nn as nn

class network(nn.Module):
    def __init__(self,  feature_extractor,numclass):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def feature_extractor(self,inputs):
        return self.feature(inputs)
    
class toplayer(nn.Module):
    def __init__(self,feature_demention,numclass):
        super(toplayer, self).__init__()
        self.fc = nn.Linear(feature_demention, numclass, bias=True)

    def forward(self, input):
        x = self.fc(input)
        return x
    
    def Incremental_learning(self,demention,numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature+demention, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]
