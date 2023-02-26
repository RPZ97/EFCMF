import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import json
import os
from typing import Any, Dict, List, Union

import numpy as np
import pydantic
import torch
import torchattacks
from BaseEvaluate import BaseEvaluate
from torch.utils.data import DataLoader
from torchattacks.attack import Attack
from tqdm import tqdm

from utils.metrics import AccuracyCls
from utils.visualization import Heatmap

class AdvExamplesParameter(pydantic.BaseModel):
    name:str = "FGSM"
    parameter:dict = {}
class ClsAdvExamplesEvaluateParameter(pydantic.BaseModel):
    inName:str = "image"
    outName:str = "cls"
    labelName:str = "label"
    advExamplesParameter: AdvExamplesParameter = AdvExamplesParameter()
    categoryList:Union[List[str],None] = None
    show:bool = False
    saveKey:str = "clsAdvExamplesEvaluate"
    saveImage:bool = False
class AttackModel(torch.nn.Module):
    def __init__(self,model:torch.nn.Module,inName:str,outName:str,labelName:str) -> None:
        super(AttackModel,self).__init__()
        self.inputDict:Dict[str,torch.Tensor] = {}
        self.inName = inName
        self.outName = outName
        self.labelName = labelName
        self.model = model
        if "BiModalPMA" in self.model._get_name():
            torch.backends.cudnn.enabled = False
            print(f"disable cudnn")
        else:
            torch.backends.cudnn.enabled = True
    def setInputDict(self,inputDict:Dict[str,torch.Tensor]):
        self.inputDict = inputDict
    def forward(self,tensor:torch.FloatTensor):
        self.inputDict[self.inName] = tensor
        return self.model(self.inputDict)[self.outName]
class ClsAdvExamplesEvaluate(BaseEvaluate):
    def __init__(self, parameter: dict, model:torch.nn.Module, logDir:str, dataLoader:DataLoader) -> None:
        super(ClsAdvExamplesEvaluate,self).__init__(parameter, model, logDir, dataLoader)
        self.parameter = ClsAdvExamplesEvaluateParameter(**parameter)
        self.logDir = logDir
        self.model = model.eval()
        self.mapModel:AttackModel = AttackModel(model,self.parameter.inName,self.parameter.outName,self.parameter.labelName) 
        self.attackModel:Attack = getattr(torchattacks,self.parameter.advExamplesParameter.name)(self.mapModel,**self.parameter.advExamplesParameter.parameter)
        self.dataLoader = dataLoader
        self.metrics = AccuracyCls({"name":self.parameter.outName})
        self.fileName = os.path.join(logDir,"evaluateResult.json")
        self.imagePath = os.path.join(logDir,"advExamples",self.parameter.advExamplesParameter.name)
        if not os.path.isdir(self.imagePath) and self.parameter.saveImage:
            os.makedirs(self.imagePath)
    def evaluate(self):
        iteration:List[Dict[str,Any]] = tqdm(self.dataLoader)
        resultList = []
        saveDict:Dict[str,Dict] = {}
        if os.path.isfile(self.fileName):
            with open(self.fileName,"r") as file:
                jsonData = json.load(file)
                saveDict.update(jsonData)
        if (self.parameter.saveKey not in saveDict.keys()):
            saveDict[self.parameter.saveKey] = {}
        if (self.parameter.advExamplesParameter.name not in saveDict[self.parameter.saveKey].keys()):
            for i,inputDict in enumerate(iteration):
                self.mapModel.setInputDict(inputDict)
                image:torch.FloatTensor = Heatmap.unNormalize(inputDict[self.parameter.inName]) 
                label = inputDict[self.parameter.labelName]
                advExamples:torch.FloatTensor = self.attackModel(image,label)
                if self.parameter.show:
                    Image.fromarray((advExamples*255).transpose(-2,-3).transpose(-1,-2)[0].detach().cpu().numpy().astype(np.uint8)).show()
                newInputDict:dict = inputDict
                newInputDict.update({self.parameter.inName:Heatmap.normalize(advExamples)})
                self.model.eval()
                output = self.model(newInputDict)
                allInfoDict = {"output":output,**newInputDict}
                result = self.metrics(allInfoDict)
                resultList.append(result)
                if self.parameter.saveImage:
                    for j,imageItem in enumerate(advExamples):
                        path = ""
                        if "path" in inputDict.keys():
                            pathData = inputDict["path"][j]
                            path = os.path.join(os.path.basename(os.path.dirname(pathData)),os.path.basename(pathData))
                        else:
                            path = f"{label[j]}/{i}-{j}.png"
                        savePath = os.path.join(self.imagePath,path)
                        if not os.path.isdir(os.path.dirname(savePath)): 
                            os.makedirs(os.path.dirname(savePath))
                        Image.fromarray((imageItem.transpose(-2,-3).transpose(-1,-2)*255).clamp(0,255).detach().cpu().numpy().astype(np.uint8)).save(savePath)
            saveDict[self.parameter.saveKey][self.parameter.advExamplesParameter.name]={"accuracy":torch.FloatTensor(resultList).mean().item()}
        with open(self.fileName,"w") as file:
            file.write(json.dumps(saveDict,indent=4))
            print(
                f"write {self.parameter.advExamplesParameter.name}: \
                {saveDict[self.parameter.saveKey][self.parameter.advExamplesParameter.name]['accuracy']} \
                to {self.fileName}")

if __name__ == "__main__":
    categoryNumber = 18
    label = torch.randint(0,categoryNumber,(10000,)).cuda()
    pred = torch.randn(10000,categoryNumber).cuda()
    for i,data in enumerate(pred):
        if i%2 ==0:
            pred[i,label[i]] = 100.
    pred = torch.softmax(pred,dim=1)