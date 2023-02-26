from typing import Callable, Dict, List, Union

import numpy as np
import torch
from torch import nn
from torch.utils import hooks as hooks


def get2dProjection(activation_batch:torch.Tensor):
    activation_batch[torch.isnan(activation_batch)] = 0
    projections = []
    reshapedActivations = activation_batch.flatten(-2,-1).transpose(-1,-2)
    reshapedActivations = reshapedActivations - reshapedActivations.mean(dim=1)
    U, S, VT = torch.linalg.svd(reshapedActivations, full_matrices=True)
    projections = torch.matmul(reshapedActivations,VT[:,0].unsqueeze(-1)) 
    return projections.view(-1,*activation_batch.shape[2:])

class ActivationsAndGradients():
    def __init__(
            self,model:nn.Module,
            targetLayers:List[nn.Module],
            reshapeTransform:Union[Callable,None]) -> None:

        self.model = model
        self.reshapeTransform = reshapeTransform
        self.gradients:List[torch.Tensor] = []
        self.activations:List[torch.Tensor] = []
        self.handles:List[Dict[str,Callable]] = [
            {
                "activation":targetLayer.register_forward_hook(self.saveActivation),
                "gradient":targetLayer.register_forward_hook(self.saveGradient)
            } 
            for targetLayer in targetLayers]
    def saveActivation(self,module,input,output:torch.Tensor):
        activation = output
        if self.reshapeTransform is not None:
            activation = self.reshapeTransform(activation)
        self.activations.append(activation.detach())
    def saveGradient(self,module,input,output:torch.Tensor):
        if not hasattr(output,"requires_grad") or not output.requires_grad:
            return None
        def _storeGrad(grad):
            if self.reshapeTransform is not None:
                grad:torch.FloatTensor = self.reshapeTransform(grad)
            self.gradients = [grad.detach()] + self.gradients
        output.register_hook(_storeGrad)
    def release(self):
        for handle in self.handles:
            for value in handle.values():
                value:hooks.RemovableHandle
                value.remove()
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

class GradCamSim():
    def __init__(
            self,
            model:torch.nn.Module,
            targetLayers:List[torch.nn.Module],
            outKey:str="cls",
            reshapeTransform: Union[Callable,None] = None,
            useGradients:bool = True
        ) -> None:
        self.model = model.eval()
        self.targetLayers = targetLayers
        self.useGradients = useGradients
        self.device = next(self.model.parameters()).device
        self.outKey = outKey
        self.activationsAndGrads = ActivationsAndGradients(self.model, targetLayers, reshapeTransform)
    def getCamWeights(
        self,
        inputDict:Dict[str,torch.Tensor],
        targetLayers:List[nn.Module],
        targets: Union[List[torch.nn.Module],None],
        activations: torch.Tensor,
        grads:torch.Tensor
        )->torch.Tensor:
        return grads.flatten(-2,-1).mean(-1)
    def forward(self,inputDict:Dict[str,torch.Tensor],targets:Union[None,List[nn.Module]],eigenSmooth:bool=False):
        outputs:Dict[str,torch.Tensor] = self.activationsAndGrads(inputDict)
        if targets == None:
            targetValues,targets = torch.max(outputs[self.outKey],dim=1)
        if self.useGradients:
            self.model.zero_grad()
            loss:torch.FloatTensor = \
                targetValues.sum()
            loss.backward(retain_graph=True)
        camPerLayer = self.computeCamPerLayer(inputDict,targets,eigenSmooth)
        return self.aggregateMultiLayers(camPerLayer)
    def computeCamPerLayer(
            self,
            inputDict:Dict[str,torch.Tensor],
            targets:Union[None,List[nn.Module]],
            eigenSmooth:bool):
        activationsList = [a.detach() for a in self.activationsAndGrads.activations]
        gradsList = [g.detach() for g in self.activationsAndGrads.gradients]
        camPerTargetLayer = []
        for i,targetLayer in enumerate(self.targetLayers):
            layerActivations = None
            layerGrads = None
            if i < len(activationsList):
                layerActivations = activationsList[i]
            if i < len(gradsList):
                layerGrads = gradsList[i]
            cam = self.getCamImage(inputDict,targetLayer,targets,layerActivations,layerGrads,eigenSmooth)
            cam = torch.relu(cam)
            cam:torch.Tensor = (cam-cam.min()+1e-7)/(cam.max()-cam.min()+1e-7)
            camPerTargetLayer.append(cam[:, None, :])
        return camPerTargetLayer
    def getCamImage(
        self,
        inputDict:Dict[str,torch.Tensor],
        targetLayer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigenSmooth: bool = False) -> np.ndarray:

        weights = self.getCamWeights(inputDict,targetLayer,targets,activations,grads)
        weightedActivations:torch.Tensor = weights[:, :, None, None] * activations
        if eigenSmooth:
            cam = get2dProjection(weightedActivations)
        else:
            cam = weightedActivations.sum(dim=1)
        return cam
    def aggregateMultiLayers(self,camPerTargetLayer:torch.Tensor)->torch.Tensor:
        camPerTargetLayer = torch.relu(torch.cat(camPerTargetLayer, dim=1)) 
        result:torch.Tensor = camPerTargetLayer.mean(dim=1)
        result = (result-result.min()+1e-7)/(result.max()-result.min()+1e-7)
        return result

    def __call__(
            self,
            inputDict:Dict[str,torch.Tensor],
            targets:Union[List[nn.Module],None]=None,
            augSmooth:bool = False,
            eigenSmooth:bool = False) -> np.ndarray:
        if augSmooth is True:
            return None
        return self.forward(inputDict,targets,eigenSmooth)
if __name__ == "__main__":
    import cv2
    import numpy as np
    from PIL import Image
    from torch.nn import functional as F
    from torchvision import models, transforms
    class ModelDict(nn.Module):
        def __init__(self) -> None:
            super(ModelDict,self).__init__()
            self.backbone = models.resnet101(pretrained=True)
        def forward(self,inputDict):
            return {"cls":self.backbone(inputDict["image"])}
    model = ModelDict().cuda()
    layer = model.backbone.layer4[-1]
    cam = GradCamSim(model,[layer])
    imagePath = "dataprocess\\CUB2002011\\images\\train\\001.Black_footed_Albatross\\Black_Footed_Albatross_0007_796138.jpg"
    imagePIL = Image.open(imagePath).convert("RGB")
    imageTensor = torch.FloatTensor(np.array(transforms.ToTensor()(imagePIL.resize(size=(256,256))) )).unsqueeze(0).cuda()
    camTensor = cam({"image":imageTensor},None).unsqueeze(1)
    size = [imagePIL.size[1], imagePIL.size[0]]
    camUpsample = F.interpolate(camTensor,size=size,mode="bilinear")
    camInt = (camUpsample.squeeze(1)[0]*255).unsqueeze(-1).int().cpu().numpy().astype(np.uint8)
    camColor = cv2.applyColorMap(camInt,cv2.COLORMAP_JET)
    cv2.imwrite("test.png",camColor)
    # Image.fromarray(camInt).convert("RGB").save("test.png")