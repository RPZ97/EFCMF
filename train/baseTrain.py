import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Union


class BaseTrain(metaclass=ABCMeta):
    def __init__(self,args,setting) -> None:
        self.args = args
        self.setting = setting
    def setSettingValue(self,setting,key:str,value:str):
        typeTransform = {
            str:lambda x:str(x),
            int:lambda x:int(x),
            float:lambda x:float(x),
            bool:lambda x:bool(x),
        }
        keyLayerSplit = key.split(".")
        valueLayer = setting
        for i,keyLayer in enumerate(keyLayerSplit):
            if i < len(keyLayerSplit)-1:
                if type(valueLayer) == dict:
                    valueLayer = valueLayer[keyLayer]
                else:
                    valueLayer = getattr(valueLayer,keyLayer)
            else:
                if type(valueLayer) == dict:
                    typeString = type(valueLayer[keyLayer])
                    valueLayer[keyLayer] = typeTransform[typeString](value)
                else:
                    typeString = type(getattr(valueLayer,keyLayer))
                    setattr(valueLayer,keyLayer,typeTransform[typeString](value))
    def replaceValue(self,target:Union[list,dict],layers:str,value:Any):
        layer = target
        layerNameList = layers.split(".")
        layerIndex = None
        for i,layerName in enumerate(layerNameList):
            if(type(layer) == dict):
                layerIndex = layerName
            elif(type(layer)== list):
                layerIndex = int(layerName)
            if i < len(layerNameList)-1:
                layer = layer[layerIndex]
            else:
                layer[layerIndex] = copy.deepcopy(value)             
                break
        return target
    @abstractmethod
    def train(self):
        raise NotImplementedError