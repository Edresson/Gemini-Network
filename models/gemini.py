"""
Gemini Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class GeminiV1(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(SpiraConvV2, self).__init__()
        self.config = config
        self.mish = Mish()
        self.fc1 = nn.Linear(self.config.model['input_dim'], self.config.model['fc1_dim'])
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.mish(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x