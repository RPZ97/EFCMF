import os
import time
from typing import Dict, List, Union

import torch
from colorama import Fore, init
from pydantic import BaseModel
from torch import nn
from transformers import BertModel

from dataloader.utils.text import getEmbeddingWeight

init(True)
class TextCNNParameter(BaseModel):
    embeddingDim:int = 100 
    embeddingWeight:str = ""
    numFilters:int = 128
    filterSizeList:List[int] = [2,3,4,5,5]
    dropout:float = 0
    freeze:bool =  False
    weight:str = ""
    bert:Union[None,str] = None
    freezeEmbedding:bool = False
    maxLength:Union[int,None] = 200
class DatasetInfo(BaseModel):
    numClasses:int 
    encodeDict:dict
class TextCNN(nn.Module):
    # 多通道textcnn
    def __init__(self,modelPara:TextCNNParameter,datasetInfo={}):
        super(TextCNN, self).__init__()
        self.datasetInfo = DatasetInfo(**datasetInfo) 
        self.parameter = TextCNNParameter(**modelPara) 
        weightInfo = None
        freeze = self.parameter.freeze
        if self.parameter.weight != "":
            weightPath = ""
            if os.path.isdir(self.parameter.weight):
                for root,_,nameList in os.walk(self.parameter.weight):
                    for name in nameList:
                        if os.path.splitext(name)[1] == ".pth":
                            weightPath = os.path.join(root,name)
            elif (os.path.isfile(self.parameter.weight)):
                weightPath = self.parameter.weight
            else:
                raise ValueError("weight is not in path")
            print(Fore.GREEN+f"load weight '{weightPath}' successfully")
            weightInfo = torch.load(weightPath)
            self.parameter = TextCNNParameter(**weightInfo["setting"]["modelSetting"]["parameter"]) 
        if self.parameter.bert == None:
            self.embedding = nn.Embedding(len(self.datasetInfo.encodeDict.keys()),self.parameter.embeddingDim)
        else: 
            self.embedding = BertModel.from_pretrained(self.parameter.bert).embeddings
            self.parameter.embeddingDim = self.embedding.word_embeddings.embedding_dim
        if self.parameter.freezeEmbedding:
            for p in self.embedding.parameters():
                p.requires_grad = False
        if self.parameter.embeddingWeight != "":
            start = time.time()
            embeddingTensor = getEmbeddingWeight(self.parameter.embeddingWeight ,self.datasetInfo.encodeDict)
            self.embedding.from_pretrained(embeddingTensor)
            end = time.time()
            print(Fore.GREEN+f"load embedding weight:{self.parameter.embeddingWeight},time:{end-start:.4f}")
        self.convs = nn.ModuleList([nn.Conv2d(1, self.parameter.numFilters, (filterSize,self.parameter.embeddingDim)) for filterSize in self.parameter.filterSizeList ])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(self.parameter.dropout)
        self.inFeatures = len(self.parameter.filterSizeList)*self.parameter.numFilters
        self.linear = nn.Linear(self.inFeatures,self.datasetInfo.numClasses)
             
        if weightInfo != None: 
            self.load_state_dict(weightInfo["model_state_dict"]) 
            print("TextCNN load weight")
        if freeze:
            for p in self.parameters():
                p.requires_grad = False 
            print("freeze All TextCNN")
        
    def forward(self, inputDict):
        x = inputDict["text"]
        DEVICE = next(self.parameters()).device
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = self.embedding(x.to(DEVICE))
        mask = torch.ones_like(x,device=DEVICE)
        if "mask" in  inputDict.keys() and inputDict["mask"] != None :
            mask =  inputDict["mask"].to(DEVICE)
            if len(mask.shape) < len(x.shape):
               mask = mask.unsqueeze(-1).expand_as(x) 
        if self.parameter.maxLength != None:
            padShape = list(x.shape)
            padShape[1] = self.parameter.maxLength - padShape[1]
            x = torch.cat([x,torch.zeros(padShape,device=DEVICE)],axis=1) 
            mask = torch.cat([mask,torch.zeros(padShape,device=DEVICE)],axis=1) 
        x = x*(mask.to(DEVICE)) 
        x = x.unsqueeze(1)
        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [torch.relu(conv(x)) for conv in self.convs]
        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [self.pool(x_item) for x_item in x]
        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        feature = torch.cat(x, 1).squeeze(-1).squeeze(-1)
        x = self.dropout(feature)
        output = self.linear(x)
        return {"cls":output}