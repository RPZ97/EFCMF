import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import json
import os
from typing import Dict, List, Union

import numpy as np
import pydantic
import torch
from BaseEvaluate import BaseEvaluate
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.visualization import Cam, Heatmap
from colorama import init, Fore
init(True)
def reshape_Transformer(tensor:torch.FloatTensor, height, width):
    '''
    不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
    height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
    比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
    height = width = 224 / 32 = 7
    '''
    result = tensor.reshape(tensor.size(0),height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getLayersByString(model,layerString:str,split="."):
    layerNameList = layerString.split(split)
    layer = getattr(model,layerNameList[0])
    for layerName in layerNameList[1:]:
        layer:torch.nn.Module = getattr(layer,layerName)
    return layer
class CamParameter(pydantic.BaseModel):
    name:str
    layers: List[str]
    reshape: Union[List[int],None] = None  
class ClassificationParameter(pydantic.BaseModel):
    show:bool = False
    name:str = "cls"
    categoryList:Union[List[str],None] = None
    layerVisualization:str = "" 
    cam:Union[CamParameter,str] = None
    pr:Union[bool,None] = None
    saveKey = "ClassificationEvaluate" 
class ClassificationEvaluate(BaseEvaluate):
    def __init__(self, parameter: dict, model, logDir:str, dataLoader) -> None:
        super(ClassificationEvaluate,self).__init__(parameter, model, logDir, dataLoader)
        self.parameter = ClassificationParameter(**parameter)
        self.logDir = logDir
        self.model.eval()
        self.outList = []
        self.labelList = [] 
        self.predList = []
        self.cam = None
        reshapeTransformer = None
        self.visualizationPath = os.path.join(self.logDir,"visualization","cam")
        if not os.path.isdir(self.visualizationPath):
            os.makedirs(self.visualizationPath)
        if self.parameter.cam !=None :
            if self.parameter.cam.reshape != None:
                reshapeTransformer = lambda x:reshape_Transformer(x,height=self.parameter.cam.reshape[0],width=self.parameter.cam.reshape[1])
            layersForModel = [getLayersByString(self.model,layer) for layer in self.parameter.cam.layers]
            self.cam:Cam.BaseCam = getattr(Cam,self.parameter.cam.name)(self.model,layersForModel,reshapeTransform=reshapeTransformer)
        self.fileName = os.path.join(logDir,"evaluateResult.json")
    def evaluate(self):
        saveDict:Dict[str,Dict] = {}
        if os.path.isfile(self.fileName):
            with open(self.fileName,"r") as file:
                jsonData = json.load(file)
                saveDict.update(jsonData)
        if self.parameter.saveKey not in saveDict.keys():
            saveDict[self.parameter.saveKey] = {}
            number = 0
            for i,data in enumerate(tqdm(self.dataLoader)):
                with torch.no_grad():
                    out:torch.FloatTensor = self.model(data)[self.parameter.name]
                    if len(out.shape) == 3:
                        out = out.mean(dim=1)
                label:torch.LongTensor = data["label"]
                if self.cam!=None:
                    predValue,pred = torch.max(torch.softmax(out,dim=1),dim=1)
                    camOut:torch.FloatTensor = self.cam(data)
                    camImageList = Heatmap.addCamToImage(camOut,data["image"])
                    for j,camImage in enumerate(camImageList):
                        camImage.save(os.path.join(
                            self.visualizationPath,str(number)+"_{}-{}-{:.4f}".format(
                                self.parameter.categoryList[label[j].item()] ,self.parameter.categoryList[pred[j].item()],predValue[j].item())+".png") )
                        number = number+1
                device = out.device
                self.outList.append(out.detach())
                self.labelList.append(label.to(device).detach())
            outCat = torch.softmax(torch.cat(self.outList),dim=1) 
            labelCat = torch.cat(self.labelList)
            labelUnique = torch.unique(labelCat).squeeze()
            categoryList = ["i" for i in range(labelCat.max().item())] if self.parameter.categoryList == None else self.parameter.categoryList
            categoryListMask = [category for i,category in enumerate(categoryList) if i in labelUnique]
            cm:torch.Tensor = __class__.ConfusionMatrixCalculate(outCat,labelCat)[labelUnique,:][:,labelUnique]
            prResult= __class__.getPRcurve(outCat,labelCat)
            __class__.plotPRcurve(prResult["recall"],prResult["precision"],categoryList=self.parameter.categoryList,savePath=self.logDir,show=self.parameter.show)
            __class__.plotConfusionMatrix(cm.detach().cpu().numpy(),savePath=self.logDir,show=self.parameter.show)
            accuracy = __class__.cmAccuracyCalculate(cm)
            recall = __class__.cmRecallCalculate(cm)
            precision = __class__.cmPrecisionCalculate(cm)
            result = {"accuracy":accuracy.item(),"recall":recall.mean().item(),"precision:":precision.mean().item()}
            result.update({category:{
                "recall":recall[i].item(),
                "precision":precision[i].item(),
                } for i,category in enumerate(categoryListMask)})
            
            saveDict[self.parameter.saveKey].update(result)
        with open(os.path.join(self.fileName),"w") as jsonFile:
            jsonFile.write(json.dumps(saveDict,indent=4))
        print(Fore.GREEN+f"write: {self.fileName}\n{self.parameter.saveKey}:"+",".join([f"{key}: {value:.5f}" for key,value in saveDict[self.parameter.saveKey].items() if (type(value)==float)]))
    @staticmethod
    @torch.no_grad()
    def accuracyCalculate(tensor,label)->torch.FloatTensor:
        conf,pred = tensor.max(dim=1)
        result = torch.eq(pred,label).float()
        return result.mean()
    @staticmethod
    @torch.no_grad()
    def cmAccuracyCalculate(cm:torch.FloatTensor)->torch.FloatTensor:
        return (torch.eye(cm.size(0),device=cm.device)*cm).sum().div(cm.sum())
    @staticmethod
    @torch.no_grad()
    def cmRecallCalculate(cm:torch.FloatTensor)->torch.FloatTensor:
        eye = (torch.eye(cm.size(-2),device=cm.device)).expand_as(cm)
        return (((eye*cm).sum(dim=-2)+1e-5).div(cm.sum(dim=-2)+1e-5))
    @staticmethod
    @torch.no_grad()
    def cmPrecisionCalculate(cm:torch.FloatTensor)->torch.FloatTensor:
        eye = (torch.eye(cm.size(-2),device=cm.device)).expand_as(cm)
        return (((eye*cm).sum(dim=-1)+1e-5).div(cm.sum(dim=-1)+1e-5))
    @staticmethod
    @torch.no_grad()
    def ConfusionMatrixCalculate(tensor:torch.FloatTensor,label:torch.LongTensor,threshold:Union[float,None]=None):
        device = tensor.device
        length = (label.max()+1).item()
        originTensor = torch.ones_like(label,device=tensor.device,dtype=torch.long)#.flatten()
        if threshold == None:
            conf,pred = tensor.max(dim=1)
            cmMatrixSize = [length,length]
        else:
            pred = (tensor>=threshold).long()
            cmMatrixSize = [tensor.size(0),length,length]
        cmIndex = (label*length+pred)#.flatten()
        cmMatrix = torch.zeros(size=cmMatrixSize,device=device,dtype=torch.long)
        cmMatrix = cmMatrix.flatten(-2)
        cmMatrix = cmMatrix.scatter_add(-1,cmIndex,originTensor)
        cmMatrix = cmMatrix.view(*cmMatrixSize)
        return cmMatrix
    @staticmethod
    @torch.no_grad()
    def getPRcurve(tensor:torch.FloatTensor,label:torch.LongTensor):
        pList,rList = [],[]
        categoryNumber = label.max()+1
        predStack = torch.stack([tensor[:,j] for j in range(categoryNumber)])
        labelStack = torch.stack([(label==j).long() for j in range(categoryNumber)])
        for i in tqdm(range(101)):
            cm = __class__.ConfusionMatrixCalculate(predStack,labelStack,threshold=i/100)
            p = __class__.cmPrecisionCalculate(cm)
            r = __class__.cmRecallCalculate(cm)
            pList.append(p),rList.append(r)
        pTensor = torch.stack(pList,dim=1)
        rTensor = torch.stack(rList,dim=1)
        return {"precision":pTensor[...,1],"recall":rTensor[...,1]}
    @torch.no_grad()
    def plotPRcurve(recall:torch.FloatTensor,precision:torch.FloatTensor,savePath="./",categoryList:Union[List[str],None]=None,show=False):
        plt.clf()
        plt.figure(figsize=(20,20))
        if categoryList == None:
            categoryList = [str(i) for i in range(precision.size(0))]
        for i,item in enumerate(zip(recall.cpu().detach().numpy(),precision.cpu().detach().numpy())) :
            plt.plot(item[0],item[1],label=categoryList[i])
        plt.xlabel("recall")
        plt.ylabel("precision")
        # plt.xticks(np.arange(0,1,recall.size(-1)))
        plt.xlim(0,1.01)
        # plt.yticks(np.arange(0,1,recall.size(-1)))
        plt.ylim(0,1.01)
        plt.legend(bbox_to_anchor=[1.1,1.12])
        plt.savefig(os.path.join(savePath,"p-rCurve.png"),bbox_inches='tight')
        if show:
            plt.show()
    @staticmethod
    def plotConfusionMatrix(matrix,categoryList:Union[list,None]=None,plotCategory:bool=False,show=False,savePath="./",yLabel="y",xLabel="x"):
        plt.clf()
        if categoryList == None:
            categoryList = [i for i in range(matrix.shape[0])]
        assert matrix.shape[0] == len(categoryList)
        categoryIndexList = [i for i in range(matrix.shape[0])]
        # plt.figure(figsize=(len(categoryList),len(categoryList)))
        fig, ax = plt.subplots(dpi=1200)
        # plt.xlabel('x',fontsize=11)
        # plt.ylabel('y',fontsize=11)
        im = ax.imshow(matrix,cmap="jet")
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        if plotCategory:
            ax.set_xticks(np.array(categoryIndexList))
            ax.set_yticks(np.array(categoryIndexList))
            ax.set_xticklabels(categoryList)
            ax.set_yticklabels(categoryList)
            plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
            for i,category in enumerate(categoryList):
                for j,category in enumerate(categoryList):
                    text  = ax.text(j,i,matrix[i,j],ha="center",va="center",color="w")
        fig.savefig(os.path.join(savePath,"confusionMatrix.png"),bbox_inches='tight')
        if show:
            plt.show() 
if __name__ == "__main__":
    categoryNumber = 18
    label = torch.randint(0,categoryNumber,(10000,)).cuda()
    pred = torch.randn(10000,categoryNumber).cuda()
    for i,data in enumerate(pred):
        if i%2 ==0:
            pred[i,label[i]] = 100.
    pred = torch.softmax(pred,dim=1)
    result = ClassificationEvaluate.getPRcurve(pred,label)
    ClassificationEvaluate.plotPRcurve(result["recall"],result["precision"])
    cmMatrix = ClassificationEvaluate.ConfusionMatrixCalculate(pred,label)
    print(cmMatrix)
    # ClassificationEvaluate.plotConfusionMatrix(cmMatrix.detach().cpu().numpy(),show=False)
    print("accuracy: {:.5f},recall: {:.5f},precision: {:.5f}".format(
        ClassificationEvaluate.cmAccuracyCalculate(cmMatrix).item(),
        ClassificationEvaluate.cmRecallCalculate(cmMatrix).mean().item(),
        ClassificationEvaluate.cmPrecisionCalculate(cmMatrix).mean().item(),
        ))