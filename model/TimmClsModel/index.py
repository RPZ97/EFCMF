import os
import sys

sys.path.append(os.getcwd())
from typing import Union

import timm
import torch
import torch.nn as nn
from pydantic import BaseModel

from model.utils import createGraphModel
from colorama import init,Fore

init(True)
class DatasetInfo(BaseModel):
    numClasses: int 
class HookLayer(BaseModel):
    name: str
    inFeatures: int
    channelMean: bool 
class TimmClsModelParameter(BaseModel):
    backbone: str = "swin_large_patch4_window12_384_in22k"
    # hookLayer: Union[HookLayer,None] = HookLayer()
    hookLayer: Union[HookLayer,None] = None
    pretrained: bool = False
    weight:Union[str,None] = None
    freeze:bool = False
    dropout:float = 0
class TimmClsModel(nn.Module):

    def __init__(self,parameter:TimmClsModelParameter,datasetInfo:DatasetInfo={}):
        super(TimmClsModel, self).__init__()
        self.parameter = TimmClsModelParameter(**parameter)
        self.datasetInfo = DatasetInfo(**datasetInfo)
        backbone:nn.Module = timm.create_model(self.parameter.backbone,pretrained=self.parameter.pretrained)
        if self.parameter.hookLayer == None:
            self.backbone = backbone
        else:
            self.backbone = createGraphModel(backbone,[self.parameter.hookLayer.name])
            self.fc = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(self.parameter.hookLayer.inFeatures,self.datasetInfo.numClasses),
            )
        self.dropout = nn.Dropout(self.parameter.dropout)
        if self.parameter.weight != None:
            weightPath = ""
            if os.path.isdir(self.parameter.weight):
                for root,_,nameList in os.walk(self.parameter.weight):
                    for name in nameList:
                        if os.path.splitext(name)[1] == ".pth":
                            weightPath = os.path.join(root,name)
            elif(os.path.isfile(self.parameter.weight)):
                weightPath = os.path.join(root,name)
            try:
                self.load_state_dict(torch.load(weightPath,map_location="cpu")["model_state_dict"]) 
                print(Fore.GREEN+f"load {self.parameter.weight} successfully")
            except:
                print(Fore.RED+f"load {self.parameter.weight} error")
        if self.parameter.freeze:
            for p in self.parameters():
                p.requires_grad = False
    def forward(self, inputDict):
        DEVICE = next(self.parameters()).device
        if self.parameter.hookLayer == None:
            x = inputDict["image"].to(DEVICE)
            out = self.backbone(x.to(DEVICE))
        else:
            x = inputDict["image"].to(DEVICE)
            hook:torch.FloatTensor = self.backbone(x.to(DEVICE))[0]
            if self.parameter.hookLayer.channelMean:
                hook = hook.mean(dim=1)
            hook = self.dropout(hook)
            out = self.fc(hook)  
        return {"cls":out} 
    def _get_name(self):
        return "_".join(self.parameter.backbone.split("_")[:2]) 
if __name__ == "__main__":
    # model = TimmClsModel({"pretrained":False},{"numClasses":200}).to("cuda" if torch.cuda.is_available() else "cpu")
    # model = TimmClsModel({"pretrained":True,"backbone":"resnet101"},{"numClasses":200}).to("cuda" if torch.cuda.is_available() else "cpu")
    # model = TimmClsModel({"pretrained":False,"backbone":"vgg16"},{"numClasses":200}).to("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClsModel({"pretrained":False,"backbone":"densenet121"},{"numClasses":200}).to("cuda" if torch.cuda.is_available() else "cpu")
    print(model)
    inputDict = {"image":torch.randn(2,3,384,384)}
    out =model(inputDict)
    for key,value in out.items():
        print(key)
        for i in value:
            print(i.shape)