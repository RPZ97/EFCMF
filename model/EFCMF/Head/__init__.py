from typing import Dict

import pydantic
import torch


class LinearHeadParameter(pydantic.BaseModel):
    inFeatures:int
class DatasetParameter(pydantic.BaseModel):
    numClasses:int
class LinearHead(torch.nn.Module):
    def __init__(self,parameter:Dict,datasetInfo:Dict) -> None:
        super(LinearHead,self).__init__()
        self.parameter = LinearHeadParameter(**parameter)
        self.datasetInfo = DatasetParameter(**datasetInfo)
        self.linear = torch.nn.Linear(in_features=self.parameter.inFeatures,out_features=self.datasetInfo.numClasses)
    def forward(self,x):
        return self.linear(x)