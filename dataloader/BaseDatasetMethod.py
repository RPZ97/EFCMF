from typing import Any, Callable, Dict, List, Union

from pydantic import BaseModel
from torch.utils.data import Dataset
from colorama import init,Fore
init()
class DatasetStructure(BaseModel):
    trainset:Any
    validset:Union[Any,None]
class DatasetInfo(BaseModel):
    dataset:DatasetStructure
    info:Dict
    collate_fn: Union[Any,None]
class ConcatDataset(Dataset):
    def __init__(self,datasetList:List[Dataset]) -> None:
        self.datasetList = datasetList
        self.length = len(self.datasetList[0])
        for dataset in datasetList:
            if self.length != len(dataset):
                raise ValueError(Fore.RED+f"length of dataset is not equal:{type(dataset)}")
    def __getitem__(self, index: int):
        returnDict = {}
        for dataset in self.datasetList:
            data:Dict[str:Any] = dataset[index]
            returnDict.update(data)
        return returnDict
    def __len__(self):
        return self.length

class CollateFnDelegate():
    def __init__(self,functionList:List[Callable]) -> Callable:
        self.functionList = functionList
    def forward(self,batchData:List[Dict]):
        dataDict:Dict[str,List] = {}
        for data in batchData:
            for key,value in data.items():
                if key not in dataDict.keys():
                    dataDict[key] = []
                dataDict[key].append(value)
        returnData = {}
        for function in self.functionList:
            returnData.update(function(dataDict))
        return returnData
    def __call__(self,batchData):
        out =self.forward(batchData)
        return out
class BaseDatasetMethod():
    def __init__(self,parameter) -> None:
        self.parameter = parameter
        self.datasetInfo = {}
        self.collate_fn = None
        self.dataset = {}
    def getDataset(self)->Dict:
        return self.dataset
    def updateDatasetInfo(self,dictData:Dict):
        self.datasetInfo.update(dictData)
    def updateDataset(self,datasetDict:Dict[str,Any]):
        self.dataset.update(datasetDict)
    def getDatasetInfo(self,key=None):
        if key == None:
            return self.datasetInfo
        else:
            return self.datasetInfo[key]
    def getDatasetAllInfo(self)->DatasetInfo:
        dataset = self.getDataset()
        assert "trainset" in dataset.keys() and dataset["trainset"] != None
        if "validset" not in dataset.keys():
            dataset["validset"] = None
        returnData = {"dataset":dataset,"info":self.datasetInfo}
        if self.collate_fn != None :
            returnData.update({"collate_fn":self.collate_fn})
        return  DatasetInfo(**returnData)
    def setCollateFn(self,function:Callable):
        self.collate_fn = function
    def getName(self):
        return self.__class__.__name__