from typing import Dict, List

import pydantic
import torch
from torch.nn import functional as F
from torch.fft import fft,ifft

def prod(numberList:List[int]):
    out = 1
    for number in numberList:
        out = out*number 
    return out
# class NeckParameter(pydantic.BaseModel):
#     inFeatureList:List[int]
class ConcatNeck(torch.nn.Module):
    def __init__(self,parameter:Dict,datasetInfo:Dict) -> None:
        super(ConcatNeck,self).__init__()
        # self.parameter = NeckParameter(**parameter)
    def forward(self,inputTensorList:List[torch.FloatTensor]):
        return torch.cat(inputTensorList,dim=1)
class AddNeck(torch.nn.Module):
    def __init__(self,parameter:Dict,datasetInfo:Dict) -> None:
        super(AddNeck,self).__init__()
        # self.parameter = NeckParameter(**parameter)
    def forward(self,inputTensorList:List[torch.FloatTensor]):
        shapeList =  [inputTensor.flatten(1).shape[-1] for inputTensor in inputTensorList]
        shapeMaxIndex = shapeList.index(max(shapeList))
        inputTensorNewList:List[torch.FloatTensor] = [inputTensor.unsqueeze(1) for inputTensor in inputTensorList]
        return sum([F.interpolate(inputTensor,size=inputTensorNewList[shapeMaxIndex].shape[2:]) 
        for inputTensor in inputTensorNewList]).squeeze(1)
class BilinearPoolNeck(torch.nn.Module):
    def __init__(self,parameter:Dict,datasetInfo:Dict) -> None:
        super(BilinearPoolNeck,self).__init__()
    def forward(self,inputTensorList:List[torch.FloatTensor]):
        assert len(inputTensorList) == 2
        feature1 = inputTensorList[0].flatten(1).unsqueeze(2)
        feature2 = inputTensorList[1].flatten(1).unsqueeze(1)
        featureFusion = torch.bmm(feature1,feature2)
        # featureFusion = F.normalize(featureFusion.flatten(1),dim=1)
        out = featureFusion.sign()*(featureFusion.abs()+1e-10).sqrt()
        out = F.normalize(out.flatten(1),dim=1,p=2)
        return out 
class MCBParameter(pydantic.BaseModel):
    inFeatureList:List[int]
    outDim:int
class MCBPoolingNeck(torch.nn.Module):
    def __init__(self,parameter:Dict,datasetInfo) -> None:
        super(MCBPoolingNeck,self).__init__()
        self.parameter = MCBParameter(**parameter)
        featureDim1,featureDim2 = self.parameter.inFeatureList[:2]
        self.d = self.parameter.outDim
        self.h,self.s = self.count_sketch_init(featureDim1,featureDim2,self.d)
        self.h0,self.h1 = self.h
        self.s0,self.s1 = self.s
    def forward(self,x:List[torch.FloatTensor]):
        feature1,feature2 = x[:2]
        DEVICE = feature1.device
        sketchFeature1 = self.count_sketch(self.d,self.h0.to(DEVICE),self.s0.to(DEVICE),feature1)
        sketchFeature2 = self.count_sketch(self.d,self.h1.to(DEVICE),self.s1.to(DEVICE),feature2)
        fftFeature1 = fft(sketchFeature1,dim=-1)
        fftFeature2 = fft(sketchFeature2,dim=-1)
        ewpFeature = torch.mul(fftFeature1,fftFeature2)
        ifftFeature = ifft(ewpFeature,dim=-1)
        mcbFeature = torch.real(ifftFeature)
        return mcbFeature
    def count_sketch_init(self,featureDim1,featureDim2, d):
        h = [torch.randint(0,d-1,size=[featureDim1]), torch.randint(0,d-1,size=[featureDim2])]
        s = [torch.floor(torch.zeros(featureDim1).uniform_(0,2))*2-1,torch.floor(torch.zeros(featureDim2).uniform_(0,2))*2-1]
        self.register_buffer('h0',h[0])
        self.register_buffer('h1',h[1])
        self.register_buffer('s0',s[0])
        self.register_buffer('s1',s[1])
        return h, s
    def count_sketch(self,d,h,s,feature):
        batchSize=feature.shape[0]
        newFeature = torch.zeros(batchSize,d,device=feature.device,requires_grad=True)
        out = newFeature.scatter_add(1,h.expand_as(feature),feature*s.expand_as(feature))
        return out
if __name__ == "__main__":
    # neck = BilinearPoolNeck({"inFeatureList":[200,200]},{})
    neck = MCBPoolingNeck({"inFeatureList":[200,400],"outDim":8000},{})
    out = neck([torch.randn(2,200*(i+1)) for i in range(2)])
    print(out.shape)