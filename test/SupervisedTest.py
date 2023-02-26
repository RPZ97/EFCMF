import os
import time
from test.BaseTest import BaseTest
from typing import Dict, List, Union

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader

import dataloader
import loss
import model as customModel
from dataloader.BaseDatasetMethod import BaseDatasetMethod
from Interface import Args
from utils import evaluate


class DatasetSetting(BaseModel):
    parameter: Dict 
    method: str    
class LossSetting(BaseModel):
    method: str 
    parameter: Dict
class ModelSetting(BaseModel):
    name: str 
    parameter: Dict 
class EvaluateSetting(BaseModel):
    setName:str = "validset"
    method: List[str]
    methodParameter: List[dict] = []
    logDir:Union[None,str] = None
class TestSetting(BaseModel):
    method: str
    loader: Dict
class Setting(BaseModel):
    modelSetting: ModelSetting
    datasetSetting: DatasetSetting
    lossSetting : LossSetting
    evaluateSetting: EvaluateSetting
    testSetting: TestSetting
class SupervisedTest(BaseTest):
    def __init__(self, args:Args, setting) -> None:
        super(SupervisedTest,self).__init__(args, setting)
        self.args = args
        self.setting:Setting = Setting(**setting)
        if args.weight != "":
            weightPara = torch.load(args.weight,map_location=torch.device('cpu'))
        if len(args.resetKeyList)>0:
            assert len(args.resetKeyList) == len(args.resetValueList)
            for key,value in zip(args.resetKeyList,args.resetValueList):
                self.setSettingValue(self.setting,key,value)
        self.device = torch.device("cuda:{}".format(0 if self.args.localDevice==-1 else args.localDevice)) if torch.cuda.is_available() != args.cpu else torch.device("cpu")
        print("device:{}".format(self.device))
        modelName = self.setting.modelSetting.name
        datasetMethod:BaseDatasetMethod = getattr(dataloader,self.setting.datasetSetting.method)(self.setting.datasetSetting.parameter)
        self.datasetInfo = datasetMethod.getDatasetAllInfo()
        model = getattr(customModel,modelName)(self.setting.modelSetting.parameter,self.datasetInfo.info).to(self.device)
        self.model:torch.nn.Module = model
        self.criterion = getattr(loss,self.setting.lossSetting.method)(self.setting.lossSetting.parameter)
        localTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        logDir =  "/".join(["log",datasetMethod.getName(),"evaluate",localTime])
        if args.weight != "":
            try:
                if args.logPath == "":
                    logDir = os.path.dirname(args.setting) 
                else:
                    logDir = args.logPath
                modelWeight = weightPara["model_state_dict"] 
                self.initialEpoch = weightPara["epoch"]+1
                self.model.load_state_dict(modelWeight)
                print("load weight successfully:",args.weight)
            except:
                print("fail to load weight:",args.weight)
        dataset = self.datasetInfo.dataset 
        evaluateSetting = self.setting.evaluateSetting
        datasetLoader = DataLoader(getattr(dataset,evaluateSetting.setName),**self.setting.testSetting.loader,
            **{key:value for key,value in {"collate_fn":self.datasetInfo.collate_fn}.items() if value!=None})
        self.evaluateMethodList:List[evaluate.BaseEvaluate] = []
        for i,method in enumerate(evaluateSetting.method):
            evaluateSetting.methodParameter.append({})
            for key in self.datasetInfo.info.keys():
                evaluateSetting.methodParameter[i][key]=self.datasetInfo.info[key]
            evaluateMethod = getattr(evaluate,method)(evaluateSetting.methodParameter[i],model,logDir,datasetLoader)
            self.evaluateMethodList.append(evaluateMethod)
    def test(self):
        for evaluateMethod in self.evaluateMethodList:
            evaluateMethod.evaluate()