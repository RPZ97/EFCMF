from abc import ABCMeta, abstractmethod


class BaseTest(metaclass=ABCMeta):
    def __init__(self,args,setting) -> None:
        self.args = args
        self.setting = setting
    def setSettingValue(self,setting,key,value):
        typeTransform = {
            type(""):lambda x:str(x),
            type(0):lambda x:int(x),
            type(0.):lambda x:float(x),
            type(True):lambda x:bool(x)
        }
        keyLayerSplit = key.split(".")
        valueLayer = setting
        for i,keyLayer in enumerate(keyLayerSplit):
            if i < len(keyLayerSplit)-1:
                if type(valueLayer) == type({}):
                    valueLayer = valueLayer[keyLayer]
                else:
                    valueLayer = getattr(valueLayer,keyLayer)
            else:
                if type(valueLayer) == type({}):
                    typeString = type(valueLayer[keyLayer])
                    valueLayer[keyLayer] = typeTransform[typeString](value)
                else:
                    typeString = type(getattr(valueLayer,keyLayer))
                    setattr(valueLayer,keyLayer,typeTransform[typeString](value))
    @abstractmethod
    def test(self):
        raise NotImplementedError