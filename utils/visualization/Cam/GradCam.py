import os
import sys

sys.path.append(os.path.dirname(__file__))
from typing import Dict, List, Union

import numpy as np
import torch
from BaseCam import BaseCam
from torch import nn


class GradCam(BaseCam):
    def getCamWeights(
        self, 
        inputDict: Dict[str, torch.Tensor], 
        targetLayers: List[nn.Module], 
        targets: Union[List[torch.nn.Module], None], 
        activations: torch.Tensor, 
        grads: torch.Tensor) -> torch.Tensor:
        return grads.flatten(-2,-1).mean(-1)

if __name__ == "__main__":
    from torchvision import models 
    class ModelDict(nn.Module):
        def __init__(self) -> None:
            super(ModelDict,self).__init__()
            self.backbone = models.resnet101(pretrained=True)
        def forward(self,inputDict):
            return {"cls":self.backbone(inputDict["image"])}
    model = ModelDict().cuda()
    cam = GradCam(model,[model.backbone.layer4[-1],model.backbone.layer3])
    targets = torch.LongTensor([3,2,1]).cuda()
    # targets = None
    imageTensor = torch.randn(3,3,224,224)
    camTensor = cam({"image":imageTensor.cuda()},targets)
    print(camTensor.shape)
    