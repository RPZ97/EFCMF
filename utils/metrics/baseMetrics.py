from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class BaseMetrics(nn.Module,metaclass=ABCMeta):
    def __init__(self,parameter={},model=None) -> None:
        super(BaseMetrics,self).__init__()
        self.logger = []
        self.parameter = parameter
        self.model = model
    def min(self)->torch.FloatTensor:
        if len(self.logger) > 0:
            return torch.FloatTensor(self.logger).min()
        else:
            return torch.FloatTensor([0])
    def max(self)->torch.FloatTensor:
        if len(self.logger) > 0:
            return max(self.logger)
        else:
            return torch.FloatTensor([0])
    def mean(self)->torch.FloatTensor:
        if len(self.logger) > 0:
            return torch.FloatTensor(self.logger).mean()
        else:
            return torch.FloatTensor([0])
    def latest(self) :
        return self.logger[-1]
    def update(self,x):
        result = self.forward(x)
        assert isinstance(result,float) == True
        self.logger.append(result)
    def reset(self):
        self.logger = []
    @abstractmethod
    def forward(self,x)->float:
        raise NotImplementedError
    def getName(self):
        return self.__class__.__name__

if __name__ == "__main__":
    class ChildMetrics(BaseMetrics):
        def __init__(self) -> None:
            pass
        def forward(self):
            pass
    a = ChildMetrics()
    print(a.getName())
