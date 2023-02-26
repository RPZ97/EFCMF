
import os
import sys

sys.path.append(os.path.dirname(__file__))
from baseLoss import BaseLoss
from torch import nn


class CrossEntropyLoss(BaseLoss):
    def __init__(self,parameter) -> None:
        super(CrossEntropyLoss,self).__init__(parameter)
        self.lossFunction = nn.CrossEntropyLoss()
    def forward(self,x):
        output = x["output"]
        cls = output["cls"]
        batchSize = cls.shape[0]
        cls = cls.view(batchSize,-1)
        label = x["label"].to(cls.device)
        loss = self.lossFunction(cls,label)
        return loss