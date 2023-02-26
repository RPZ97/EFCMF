# 环境库
import json
import os
import sys

from dataloader.BaseDatasetMethod import BaseDatasetMethod

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
# yaml 文件 setting 约束方法
from typing import Dict, List, Union

import torch
import yaml
from pydantic import BaseModel
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# 日志库
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 项目文件
import dataloader
import loss
import model as customModel
from Interface import Args
from train import learningReteAdjust
from train.baseTrain import BaseTrain
from utils import metrics


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
class SaveMethod(BaseModel):
    setName: Union[str,None] = "train" # 如train.loss
    calName: Union[str,int] = "loss" # 如train.loss
    minimize:bool = True

class Setting(BaseModel):
    modelSetting: ModelSetting
    datasetSetting: DatasetSetting
    optimizerSetting: OptimizerSetting
    lossSetting : LossSetting
    trainSetting: TrainSetting
    validSetting : ValidSetting
    saveMethod: SaveMethod
    epochs: int
    test:bool = False
# -----------------
class SupervisedLearning(BaseTrain):
    def __init__(self, args:Args, setting) -> None:
        super(SupervisedLearning,self).__init__(args, setting)
        self.args = args
        self.setting:Setting = Setting(**setting)
        self.initialEpoch = 0
        if args.weight != "":
            weightPara = torch.load(args.weight,map_location=torch.device('cpu'))
            self.setting:Setting = Setting(**weightPara["setting"]) 
        if len(args.resetKeyList)>0:
            assert len(args.resetKeyList) == len(args.resetValueList)
            for key,value in zip(args.resetKeyList,args.resetValueList):
                self.setSettingValue(self.setting,key,value)
        self.device = torch.device("cuda:{}".format(0 if self.args.localDevice==-1 else args.localDevice)) if torch.cuda.is_available() != args.cpu else torch.device("cpu")
        if self.args.distributed:
            dist.init_process_group(backend=self.args.distBackend, rank=self.args.rank, world_size=self.args.worldSize,init_method=self.args.initMethod)
            torch.cuda.set_device(self.device)
            try: 
                self.setting.trainSetting.loader["batch_size"] = int(self.setting.trainSetting.loader["batch_size"]/self.args.worldSize)  
                self.setting.validSetting.loader["batch_size"] = int(self.setting.validSetting.loader["batch_size"]/self.args.worldSize)
                self.setting.trainSetting.loader["shuffle"] = False
                self.setting.validSetting.loader["shuffle"] = False
            except:
                pass
        print("device:{}".format(self.device))
        modelName = self.setting.modelSetting.name
        datasetMethod:BaseDatasetMethod = getattr(dataloader,self.setting.datasetSetting.method)(self.setting.datasetSetting.parameter)
        self.datasetInfo = datasetMethod.getDatasetAllInfo()
        model:torch.nn.Module = getattr(customModel,modelName)(self.setting.modelSetting.parameter,self.datasetInfo.info).to(self.device)
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True,broadcast_buffers=False,device_ids=[self.args.localDevice],output_device=self.device)
            self.model._get_name = model._get_name
        else:
            self.model = model
        self.optimizer:torch.optim.Optimizer = getattr(torch.optim,self.setting.optimizerSetting.name)(self.model.parameters(),**self.setting.optimizerSetting.parameter)
        self.scaler = None 
        if self.setting.trainSetting.useScaler:
            self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = getattr(loss,self.setting.lossSetting.method)(self.setting.lossSetting.parameter)
        if args.weight != "":
            try:
                modelWeight = weightPara["model_state_dict"] 
                optimizerWeight = weightPara["optimizer_state_dict"] 
                self.initialEpoch = weightPara["epoch"]+1
                self.model.load_state_dict(modelWeight)
                self.optimizer.load_state_dict(optimizerWeight)
                print("load weight successfully:",args.weight)
            except:
                print("fail to load weight:",args.weight)
        if self.args.rank == 0:
            print(json.dumps(self.setting.dict(),indent=4) )
            localTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            logDir =  "/".join(["log","train",datasetMethod.getName(),self.model._get_name(),localTime]) if (self.args.logPath == "") else os.path.join(self.args.logPath,localTime)
            self.weightPath = f"{logDir}/weight" 

            if not os.path.isdir(self.weightPath):
                os.makedirs(self.weightPath)
            with open(os.path.join(logDir,os.path.basename(args.setting)),"w",encoding="utf-8") as settingFile:
                saveSetting = {}
                saveSetting.update(setting)
                saveSetting.update(self.setting.dict())
                yaml.dump(saveSetting,stream=settingFile,allow_unicode=True,indent=4)
            self.tbWriter = SummaryWriter(log_dir=logDir,comment="")
        self.trainMetrics:List[metrics.BaseMetrics] = self.getMetricsList(self.setting.trainSetting.metrics,self.setting.trainSetting.metricsParameter)
        self.validMetrics:List[metrics.BaseMetrics] = self.getMetricsList(self.setting.validSetting.metrics,self.setting.validSetting.metricsParameter)
    def getMetricsList(self,metricsNameList,metricsParameterList=[]):
        metricsList = []
        for i,metricsName in  enumerate(metricsNameList):
            try:
                metricsParameter = metricsParameterList[i]
            except:
                metricsParameter = {}
            metricsList.append(getattr(metrics,metricsName)(metricsParameter,self.model).to(self.device))
        return metricsList
    def reduce_mean(self,tensor:torch.FloatTensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.worldSize
        return rt
    @torch.no_grad()
    def validate(self,loader:tqdm):
        self.model.eval()
        lossList = []
        minLoss = None
        for step,data in enumerate(loader):
            allData = {}
            output = self.model(data)
            allData.update({"output":output})
            allData.update(data)
            loss:torch.FloatTensor = self.criterion(allData)/self.setting.trainSetting.optimizerUpdateStep
            for metricsItem in self.validMetrics:
                metricsItem.update(allData)
            if self.args.distributed:
                dist.barrier()
                loss = self.reduce_mean(loss)
                if step == 0 :
                    minLoss = loss.item()
                else:
                    if loss.item() <minLoss :
                        minLoss = loss.item()
            else:
                if step == 0 :
                    minLoss = loss.item()
                else:
                    if loss.item() <minLoss :
                        minLoss = loss.item()
            loader.set_description(f"loss:{loss:4f} minLoss:{minLoss:4f}")
            lossList.append(loss.detach().cpu())
            if self.setting.test == True:
                break
        return  {"loss":lossList}

    def trainEpoch(self,loader:tqdm,epoch:int,decayMethod:learningReteAdjust.BaseLearningRateAdjust):
        self.model.train()
        lossList = []
        minLoss = None
        for step,data in enumerate(loader):
            allData = {}
            decayMethod.adjust(self.optimizer,epoch,step)
            with autocast(enabled=self.setting.trainSetting.useScaler):
                output = self.model(data)
                allData.update({"output":output})
                allData.update(data)
                loss:torch.FloatTensor = self.criterion(allData)/self.setting.trainSetting.optimizerUpdateStep
            if self.scaler !=None :
                self.scaler.scale(loss).backward()
                if (1+step)%self.setting.trainSetting.optimizerUpdateStep == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (1+step)%self.setting.trainSetting.optimizerUpdateStep == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            for metricsItem in self.trainMetrics:
                metricsItem.update(allData)
            if self.args.distributed:
                dist.barrier()
                loss:torch.FloatTensor = self.reduce_mean(loss)
                if step == 0 :
                    minLoss = loss.item()
                else:
                    if loss.item() <minLoss :
                        minLoss = loss.item()
            else:
                if step == 0 :
                    minLoss = loss.item()
                else:
                    if loss.item() <minLoss :
                        minLoss = loss.item()
            loader.set_description(f"loss:{loss.item():4f} minLoss:{minLoss:4f}")
            lossList.append(loss.detach().cpu())
            if self.setting.test == True:
                break
        return  {"loss":lossList}
    def train(self):
        epochInfo = {"train":{"loss":[]},"valid":{"loss":[]}}
        epochInfo["train"].update({metricsItem.getName():[] for metricsItem in self.trainMetrics})
        epochInfo["valid"].update({metricsItem.getName():[] for metricsItem  in self.validMetrics})
        epochIteration = tqdm(range(self.initialEpoch,self.setting.epochs))
        dataset = self.datasetInfo.dataset 
        trainset = dataset.trainset
        valset = dataset.validset
        sampler = DistributedSampler(trainset) if self.args.distributed else None
        trainLoader = DataLoader(trainset,**self.setting.trainSetting.loader,
            **{key:value for key,value in {"collate_fn":self.datasetInfo.collate_fn}.items() if value!=None},
            **{key:value for key,value in {"sampler":sampler}.items() if self.args.distributed})
        decayMethod:learningReteAdjust.BaseLearningRateAdjust = getattr(learningReteAdjust,self.setting.trainSetting.decayMethod.name)(
            self.setting.epochs,len(trainLoader),self.setting.optimizerSetting.parameter["lr"],self.setting.trainSetting.decayMethod.parameter)
        for epoch in epochIteration:
            iterationTrain =tqdm(trainLoader,desc="train",ascii=True)
            trainInfo = self.trainEpoch(iterationTrain,epoch,decayMethod)
            trainLoss = torch.stack(trainInfo["loss"])
            trainLoss = trainLoss[~torch.isnan(trainLoss)]
            trainLossMean = trainLoss.mean()
            epochInfo["train"]["loss"].append(trainLossMean)
            if self.args.rank == 0:
                self.tbWriter.add_scalar("train/loss",trainLossMean,epoch)
            writeString = f"{epoch:4d}.train result: loss={trainLossMean:.5f} "
            for metricsItem in self.trainMetrics:
                metricsValue = metricsItem.mean()
                if self.args.distributed:
                    dist.barrier()
                    metricsValue = self.reduce_mean(metricsValue.to(self.device))
                metricsName = metricsItem.getName()
                epochInfo["train"][metricsName].append(metricsValue)
                writeString += f" {metricsName}:{metricsValue:.5f}"
                if self.args.rank == 0:
                    self.tbWriter.add_scalar("train/"+metricsName,metricsValue.item(),epoch)
                metricsItem.reset()
            tqdm.write(writeString)
            validating = (self.setting.validSetting.interval!=None) and ((epoch+1)%self.setting.validSetting.interval==0)
            if (valset!=None) and validating: 
                sampler = DistributedSampler(valset) if self.args.distributed else None
                valLoader = DataLoader(valset,**self.setting.validSetting.loader,
                    **{key:value for key,value in{"collate_fn":self.datasetInfo.collate_fn}.items() if value!=None},
                    **{key:value for key,value in {"sampler":sampler}.items() if self.args.distributed})
                iterationVal =tqdm(valLoader,desc="validate",ascii=True)
                validInfo = self.validate(iterationVal)
                validLoss = torch.stack(validInfo["loss"])
                validLoss = validLoss[~torch.isnan(validLoss)]
                validLossMean = validLoss.mean()
                epochInfo["valid"]["loss"].append(validLossMean)
                if self.args.rank == 0:
                    self.tbWriter.add_scalar("valid/loss",validLossMean,epoch)
                validString = f"{epoch:4d}.valid result: loss:{validLossMean:.5f}"
                for metricsItem in self.validMetrics:
                    metricsValue = metricsItem.mean()
                    if self.args.distributed:
                        dist.barrier()
                        metricsValue = self.reduce_mean(metricsValue.to(self.device))
                    metricsName = metricsItem.getName()
                    epochInfo["valid"][metricsName].append(metricsValue)
                    validString += f" {metricsName}:{metricsValue.item():.5f}"
                    if self.args.rank == 0:
                        self.tbWriter.add_scalar("valid/"+metricsName,metricsValue.item(),epoch)
                    metricsItem.reset()
                tqdm.write(validString)
            if (self.setting.saveMethod.setName != None) and (self.setting.saveMethod.calName!=None):
                save = False
                saveInfo = "" 
                selectMethod = (lambda infoList:torch.stack(infoList).min()) if self.setting.saveMethod.minimize else (lambda infoList:torch.stack(infoList).max())
                setName = self.setting.saveMethod.setName
                calName = self.setting.saveMethod.calName
                if validating:
                    if epochInfo[setName][calName][-1] == selectMethod(epochInfo[setName][calName]):
                        save = True
                        saveInfo += f"{setName}.{calName}_{epochInfo[setName][calName][-1].item():.4f}"
                if save and self.args.rank == 0:
                    weightFilePath = os.path.join(self.weightPath,"_".join([self.model._get_name(),str(epoch),saveInfo])+".pth")
                    weightDeleteNameList = [self.weightPath+"/"+name for name in os.listdir(self.weightPath)] 
                    torch.save({
                        "epoch":epoch,
                        "setting":self.setting.dict(),
                        "model_state_dict":self.model.state_dict(),
                        "optimizer_state_dict":self.optimizer.state_dict(),
                        },
                        weightFilePath)
                    for deleteName in weightDeleteNameList:
                        os.remove(deleteName)
                    tqdm.write("save weight: "+weightFilePath)