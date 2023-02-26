from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

unNormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

def addCamToImage(camTensor:torch.FloatTensor,imageTensor:torch.FloatTensor)->List[Image.Image]:
    addImageList = []
    if len(camTensor.shape) == 3:
        camTensor = camTensor.unsqueeze(1)
    imageTensorUnNorm:List[torch.FloatTensor] = unNormalize(imageTensor)
    for index in range(camTensor.size(0)):
        camInterpolate:torch.FloatTensor = F.interpolate(camTensor[index:index+1],size=imageTensor.shape[-2:],mode="bilinear")
        camScale = ((camInterpolate-camInterpolate.min()+1e-7)*255/(camInterpolate.max()-camInterpolate.min()+1e-7)).clamp(0,255).int()
        camHeatmap = cv2.applyColorMap(camScale.detach().cpu().numpy().astype(np.uint8)[0,0],cv2.COLORMAP_JET) 
        camHeatmap = cv2.cvtColor(camHeatmap, cv2.COLOR_BGR2RGB)
        imageScale = imageTensorUnNorm[index].transpose(0,1).transpose(1,2)
        imageNormInverse = ((imageScale)*255).clamp(0,255).int().cpu().detach().numpy().astype(np.uint8)
        imageNumpy = cv2.cvtColor(imageNormInverse, cv2.COLOR_BGR2RGB)
        addImage = cv2.addWeighted(imageNumpy,0.5,camHeatmap,0.5,0)
        addImageList.append(Image.fromarray(addImage))
    return  addImageList