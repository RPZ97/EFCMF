from typing import Any, Dict, List, Union

from pydantic import BaseModel


class Args(BaseModel):
    mode:str
    cpu:bool
    setting:str
    logPath:str = ""
    weight:str
    distributed:bool
    distBackend:str
    rank:int
    localDevice:int
    worldSize:int
    initMethod:str
    resetKeyList:List[str]
    resetValueList:List[Any]

class DatasetSetting(BaseModel):
    parameter: Dict 
    method: str    
class ModelSetting(BaseModel):
    name: str 
    parameter: Dict 
class OptimizerSetting(BaseModel):
    name: str 
    parameter: Dict = {"lr":0.0001}
class LossSetting(BaseModel):
    method: str 
    parameter: Dict
class DecaySetting(BaseModel):
    name:str = "Default"
    parameter:dict = {}
class TrainSetting(BaseModel):
    method: str
    metrics: List
    metricsParameter: List
    loader: Dict
    useScaler:bool = True
    optimizerUpdateStep:int = 1
    decayMethod:DecaySetting = DecaySetting()
class ValidSetting(BaseModel):
    metrics: List
    metricsParameter: List
    loader: Dict
    interval: Union[int,None] = 1
class SearchSetting(BaseModel):
    parameter:List[Dict[str,Any]]
    minimize:bool = True
    searchNums:int = 20
    setName: Union[str,None] = "train" # 如train.loss
    calName: Union[str,None] = "loss" # 如train.loss
    minimize: bool = True
    saveWeight:bool = True
    interval: Union[int,None] = 1
class SettingForParameterSearch(BaseModel):
    modelSetting: ModelSetting
    datasetSetting: DatasetSetting 
    optimizerSetting: OptimizerSetting
    lossSetting : LossSetting 
    trainSetting: TrainSetting 
    validSetting : ValidSetting
    searchSetting:SearchSetting
    epochs: int
    test:bool = False