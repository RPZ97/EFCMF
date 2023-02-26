import os
from abc import ABCMeta, abstractmethod

from torch import nn


class BaseEvaluate(metaclass=ABCMeta):
    def __init__(self,parameter:dict,model,logDir,dataLoader) -> None:
        self.parameter = parameter 
        self.model:nn.Module = model 
        self.logDir = logDir 
        self.dataLoader = dataLoader 
        if not os.path.isdir(logDir) :
            os.mkdir(logDir)
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError