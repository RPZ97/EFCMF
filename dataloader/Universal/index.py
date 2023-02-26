import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from typing import Callable, Dict, List, Union

import pydantic
from torch.utils.data import Dataset

from dataloader.BaseDatasetMethod import (BaseDatasetMethod, CollateFnDelegate,
                                          ConcatDataset)

from . import getCollateFn, getDatasetInfo, getDatasetLoader


class DatasetItem(pydantic.BaseModel):
    setName:str
    name:str 
    parameter:Dict = {}

class InfoLoaderItem(pydantic.BaseModel):
    setName:Union[str,None] = None
    name:str 
    parameter:Dict = {}

class DatasetMethodParameter(pydantic.BaseModel):
    name:Union[str,None] = None
    infoLoaderList: List[InfoLoaderItem] = []
    datasetList: List[DatasetItem] 
    collateFn: Union[List[str],None] = None
class UniversalDatasetMethod(BaseDatasetMethod):
    def __init__(self, parameter={}) -> None:
        super(UniversalDatasetMethod,self).__init__(parameter)
        self.parameter = DatasetMethodParameter(**parameter)
        datasetDict:Dict[str,List] = {}
        datasetInfo:Dict[str,dict] = {}
        for infoLoaderItem in self.parameter.infoLoaderList:
            infoLoader:Callable = getattr(getDatasetInfo,infoLoaderItem.name)
            info:Dict = infoLoader(
                **infoLoaderItem.parameter)
            if (infoLoaderItem.setName!=None):
                if(infoLoaderItem.setName not in datasetInfo.keys()):
                    datasetInfo[infoLoaderItem.setName] = info
                else:
                    datasetInfo[infoLoaderItem.setName].update(info)
            else:
                self.updateDatasetInfo(info)
        for datasetItem in self.parameter.datasetList:
            inputInfo = {}
            inputInfo.update(datasetItem.parameter)
            inputInfo.update(self.getDatasetInfo())
            if datasetItem.setName in datasetInfo.keys():
                inputInfo.update(datasetInfo[datasetItem.setName])
            dataset:Dataset = getattr(getDatasetLoader,datasetItem.name)(
                **inputInfo
                )
            if datasetItem.setName not in datasetDict.keys():
                datasetDict[datasetItem.setName] = []
            datasetDict[datasetItem.setName].append(dataset)
        self.updateDataset({key:ConcatDataset(value) for key,value in datasetDict.items()})    
        if self.parameter.collateFn != None:
            collateFn = CollateFnDelegate([getattr(getCollateFn,collateFn) for collateFn in self.parameter.collateFn])
            self.setCollateFn(collateFn)
    def getName(self):
        if self.parameter.name == None:
            return __class__.__name__
        else:
            return self.parameter.name
if __name__ == "__main__":
    datasetMethod = UniversalDatasetMethod({
        "infoLoaderList":[
            {
                "setName":"trainset","name":"loadClsImageInfo",
                "parameter":{"imagePath":"dataprocess/Flower102/images/train"}
            },
            {
                "name":"loadClsCategoryList",
                "parameter":{"path":"dataprocess/Flower102/images/train"}
            }
        ],
        "datasetList":[
            {
                "setName":"trainset","name":"ImageDataset","parameter":{}
            }
        ]
    })
    print(datasetMethod.getDatasetInfo())