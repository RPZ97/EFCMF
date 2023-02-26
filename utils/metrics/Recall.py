# from utils.metrics import BaseMetrics
import os
import sys

import torch

sys.path.append(os.path.dirname(__file__))
from baseMetrics import BaseMetrics
from pydantic import BaseModel


class RecallClsParameter(BaseModel):
    name:str = "cls"
class RecallCls(BaseMetrics):
    def __init__(self,parameter={},model=None) -> None:
        super(RecallCls,self).__init__(parameter)
        self.parameter = RecallClsParameter(**parameter)
    @torch.no_grad()
    def forward(self,x):
        output, labels = x["output"][self.parameter.name],x["label"]
        if len(labels.shape)>1:
            labels = labels.to(output.device).view(labels.shape[0],-1).max(1)[1]
        else:
            labels = labels.long()
        preds = torch.softmax(output.view(output.shape[0],-1),dim=1).max(1)[1].type_as(labels)
        correct = preds.eq(labels).float()
        return correct.mean().item()
    def getName(self):
        if self.parameter.name == "cls":
            return self.__class__.__name__
        else: 
            return "-".join(["Recall",self.parameter.name]) 
if __name__ == "__main__":
    a = RecallCls({})
    for i in range(10):
        data = {
            "output":{a.parameter.name:torch.FloatTensor([[0.1,0.9,0.2],[0.2,0.8,0.3]])},
            "label":torch.LongTensor([1,1])}
        print(data["label"].shape)
        a.update(data)
    print(a.mean())