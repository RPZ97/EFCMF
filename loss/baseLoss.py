from torch import nn 
from abc import abstractmethod,ABCMeta

class BaseLoss(nn.Module,metaclass=ABCMeta):
    def __init__(self,parameter) -> None:
        super(BaseLoss,self).__init__()
        self.lossParameter = parameter

    @abstractmethod
    def forward(self,x):
        raise NotImplementedError