import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.resnet import resnet18, resnet50
from itertools import permutations
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

NT = 2



def count_occurrences(list_of_lists):
    counts = Counter(tuple(sublist) for sublist in list_of_lists)
    counts = [(list(sublist), count) for sublist, count in counts.items()]
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    return counts


class SimCLR(nn.Module):
    def __init__(self, backbone, reduce):
        super(SimCLR, self).__init__()
        
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.enc = resnet18()
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.enc = resnet50()
            self.feature_dim = 2048
        else:
            raise NotImplementedError  
            
        
        if reduce:            
            self.enc.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.enc.maxpool = nn.Identity()
                
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, affine=False)
        )
        
        # self.projector = nn.Sequential(
        #     nn.Linear(self.feature_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128),
        # )

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
    
def build_resnet18(reduce):
    return SimCLR('resnet18', reduce)

def build_resnet50(reduce):
    return SimCLR('resnet50', reduce)