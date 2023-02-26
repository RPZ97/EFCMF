from abc import abstractmethod,ABCMeta
import numpy as np 
import math
from typing import Union,Dict
from pydantic import BaseModel

class BaseLearningRateAdjust(metaclass=ABCMeta):
    def __init__(self,maxEpochs:int,maxIterations:int,lr:float,parameter:dict={}) -> None:
        self.parameter = parameter
        self.maxEpochs = maxEpochs
        self.maxIterations  = maxIterations
        self.lr = lr
    @abstractmethod
    def adjust(self,optimizer,epoch,step):
        raise NotImplementedError

class Default(BaseLearningRateAdjust):
    def adjust(self, optimizer, epoch, step):
        #不调整
        pass

class CosineDecayParameter(BaseModel): 
    warmupIterations:int = 800
    decayType:int = 1

class CosineDecay(BaseLearningRateAdjust):
    
    def __init__(self, maxEpochs:int, maxIterations:int,lr:float,parameter:dict={}) -> None:
        self.maxEpochs = maxEpochs
        self.maxIterations = maxIterations
        self.lr = lr
        self.eps = 1e-9
        self.parameter = CosineDecayParameter(**parameter)
        self.schedule = self.getCosineDecaySchedule(self.parameter.decayType)

    def getCosineDecaySchedule(self,decayType: int = 1):
        totalIterations = self.maxEpochs * self.maxIterations
        iterationsNoWarmup = np.arange(totalIterations - self.parameter.warmupIterations)

        if decayType == 1:
            schedule = np.array([self.eps + 0.5 * (self.lr - self.eps) * (1 + math.cos(math.pi * t / totalIterations)) for t in iterationsNoWarmup])
        elif decayType == 2:
            schedule = self.lr * np.array([math.cos(7*math.pi*t / (16*totalIterations)) for t in iterationsNoWarmup])
        else:
            raise ValueError("Not support this deccay type")
        
        if self.parameter.warmupIterations > 0:
            warmupSchedule = np.linspace(self.eps, self.lr, self.parameter.warmupIterations)
            schedule = np.concatenate((warmupSchedule, schedule))

        return schedule
    def adjust(self, optimizer, epoch, step):
        iterations = epoch*self.maxIterations + step
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.schedule[iterations]