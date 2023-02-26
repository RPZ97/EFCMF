import os
import random
import sys
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
import Head
import ImageClsModel
import Neck
import TextClsModel


def createGraphModel(model:nn.Module,nodes:Union[Dict[str,str],List[str],None]):
    def getHookOut(out:dict,name):
        def hook(model:nn.Module,input,output):
            out[name] = output
        return hook
    class GraphModel(nn.Module):
        def __init__(self) -> None:
            super(GraphModel,self).__init__()
            self.registerModel = model
            self.outs = {}
            if type(nodes) == type({}):
                nodeIter = nodes.keys()
            elif type(nodes) == type([]):
                nodeIter = nodes
            if nodes != None:
                for nodeGetName in nodeIter:
                    if nodeGetName == None:
                        continue
                    nodeGetNameList = nodeGetName.split(".")
                    layer = self.registerModel
                    for nodeRegisterName in nodeGetNameList:
                        layer:nn.Module = getattr(layer,nodeRegisterName)
                    layer.register_forward_hook(getHookOut(self.outs,nodeGetName))
        def forward(self,x):
            out = model(x)
            if type(nodes) == type({}):
                return {nodeOutName:self.outs[nodeGetName] for nodeGetName,nodeOutName in nodes.items()}
            elif type(nodes) == type([]):
                return [self.outs[node] for node in nodes]
            else:
                return out
        def _get_name(self):
            return f"{self.registerModel._get_name()}"
    return GraphModel()

class DatasetInfo(BaseModel):
    numClasses: int 
    encodeDict: Union[dict,None] = None 
    graph: Union[Any,None] = None

class VisionParameter(BaseModel):
    name:str = "TimmClsModel"
    parameter:Dict = {
        "backbone":"resnet101",
        "hookLayer":"global_pool",
        "inFeatures":2048}
    featureLayer:str = "backbone.registerModel.norm"
    inFeatures:int
class LanguageParameter(BaseModel):
    name:str = "TextCNN"
    parameter:Dict = {}
    featureLayer:Union[str,None] = "dropout"
    inFeatures:int
class NeckParameter(BaseModel):
    name:str = "ConcatNeck"
    parameter:Dict = {}
class HeadParameter(BaseModel):
    name:str = "fc"
    parameter:Dict = {}
class EMCMFParameter(BaseModel):
    visionParameter:VisionParameter
    languageParameter:LanguageParameter
    neckParameter:NeckParameter = NeckParameter()
    headParameter:HeadParameter = HeadParameter()
    dropout:float = 0.5
    dropFeature:bool = False
    dropInput:bool = False
    p:float = 0.8 #模态失活概率P
    q:float = 0.2 #视觉模态完整性概率Q
    pseudoRandom:bool = False
    dropVisionFeature:bool = False 
    dropLanguageFeature:bool = False
class EFCMF(nn.Module):
    def __init__(self, parameter, datasetInfo) -> None:
        super(EFCMF,self).__init__()
        self.parameter = EMCMFParameter(**parameter)
        self.datasetInfo = DatasetInfo(**datasetInfo)
        visionModel = getattr(ImageClsModel,self.parameter.visionParameter.name)(self.parameter.visionParameter.parameter,self.datasetInfo.dict())
        languageModel = getattr(TextClsModel,self.parameter.languageParameter.name)(self.parameter.languageParameter.parameter,self.datasetInfo.dict())
        self.datasetInfo = DatasetInfo(**datasetInfo)
        self.visionModel = createGraphModel(visionModel,[self.parameter.visionParameter.featureLayer])
        self.languageModel = createGraphModel(languageModel,[self.parameter.languageParameter.featureLayer])
        self.neck:nn.Module = getattr(Neck,self.parameter.neckParameter.name)(self.parameter.neckParameter.parameter,self.datasetInfo.dict())
        if self.parameter.headParameter.name == "fc":
            # 为了兼容老版本
            self.fc = nn.Linear(self.parameter.visionParameter.inFeatures+self.parameter.languageParameter.inFeatures,self.datasetInfo.numClasses)
            self.head = self.fc
        else:
            self.headModel:nn.Module = getattr(Head,self.parameter.headParameter.name)(self.parameter.headParameter.parameter,self.datasetInfo.dict())
            self.head = self.headModel
        self.dropout = nn.Dropout(self.parameter.dropout) 
    def getFeatureDropMask(self,feature1:torch.FloatTensor,feature2:torch.FloatTensor,activate=False):
        visionMask,languageMask = torch.ones_like(feature1,device=feature1.device),\
            torch.ones_like(feature2,device=feature2.device)
        if self.training and activate:
            batchSize = visionMask.size(0)
            if self.parameter.pseudoRandom:
                numDrop = min(int(batchSize*self.parameter.p),batchSize)
                numBalance = min(int(batchSize*self.parameter.q*self.parameter.p),numDrop)
                balanceList = [[1 if i<numBalance else 0,0 if i <numBalance else 1] 
                    if (i<numDrop) else [1,1] for i in range(batchSize) ]
                random.shuffle(balanceList)
            else:
                randomList = [[random.uniform(0,1),random.uniform(0,1)] for i in range (batchSize)]
                balanceList = [[int(randomCouple[1]<=self.parameter.q),int(randomCouple[1]>self.parameter.q)] 
                    if randomCouple[0]<self.parameter.p else [1,1] 
                    for randomCouple in randomList]
            balanceTensor = torch.LongTensor(balanceList).float().to(feature1.device)
            balanceTensor1,balanceTensor2 = balanceTensor[:,0:1],balanceTensor[:,1:2]
            for i in range(len(visionMask.shape)-len(balanceTensor1.shape)):
                balanceTensor1 = balanceTensor1.unsqueeze(-1)
            for i in range(len(languageMask.shape)-len(balanceTensor2.shape)):
                balanceTensor2 = balanceTensor2.unsqueeze(-1)
            visionMask = visionMask*balanceTensor1
            languageMask = languageMask*balanceTensor2
            visionMask = visionMask*balanceTensor1
            languageMask = languageMask*balanceTensor2
        if self.parameter.dropVisionFeature:
            visionMask = torch.zeros_like(feature1,device=feature1.device)
        if self.parameter.dropLanguageFeature:
            languageMask = torch.zeros_like(feature2,device=feature2.device)
        return visionMask,languageMask
    def forward(self,inputDict:Dict[str,Any]):
        DEVICE = next(self.parameters()).device
        image = inputDict["image"]
        mask = inputDict["mask"]
        imageMask,maskMask = self.getFeatureDropMask(image,mask,self.parameter.dropInput)
        inputDict.update({"image":image*imageMask,"mask":mask*maskMask})
        visionFeature:torch.FloatTensor = self.visionModel(inputDict)[0]
        if len(visionFeature.shape)==3: visionFeature = visionFeature.mean(dim=1)
        languageFeature:torch.FloatTensor = self.languageModel(inputDict)[0]
        visionFeature = visionFeature.flatten(1)
        languageFeature = languageFeature.flatten(1)
        visionMask,languageMask = self.getFeatureDropMask(visionFeature,languageFeature,self.parameter.dropFeature)
        fusionFeature = self.neck([visionFeature*visionMask,languageFeature*languageMask]).flatten(1)
        fusionFeature = self.dropout(fusionFeature)
        out = self.head(fusionFeature)
        return {"cls":out}
    def _get_name(self):
        if self.parameter.dropFeature:
            return f"{self.__class__.__name__}-{self.visionModel._get_name()}-{self.languageModel._get_name()}-{self.neck._get_name()}-p_{self.parameter.p:.2f}-q_{self.parameter.q:.2f}"
        else:
            return f"{self.__class__.__name__}-{self.visionModel._get_name()}-{self.languageModel._get_name()}-{self.neck._get_name()}"
if __name__ == "__main__":
    model = EFCMF(
        {
            "visionParameter":{
                "pretrained":False,
                },
            "languageParameter":{},
            "dropFeature":True,"q":0.2},
        {"numClasses":200,"encodeDict":{str(i):i for i in range(50)}}
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    print(model._get_name())
    batchSize = 16
    inputDict = {"image":torch.randn(batchSize,3,384,384),"text":torch.randint(0,49,(batchSize,200))}
    out =model(inputDict)
    for key,value in out.items():
        print(key)
        for i in value:
            print(i.shape)