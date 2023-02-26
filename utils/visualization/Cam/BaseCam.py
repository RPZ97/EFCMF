from typing import Callable, Dict, List, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import hooks as hooks


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

class BaseCam():
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
        raise Exception("Not Implemented")
    def forward(self,inputDict:Dict[str,torch.Tensor],targets:Union[None,torch.LongTensor],eigenSmooth:bool=False):
        outputs:Dict[str,torch.Tensor] = self.activationsAndGrads(inputDict)
        if targets == None:
            targetValues,targets = torch.max(outputs[self.outKey],dim=1)
        else:
            if len(targets.shape) < len(outputs[self.outKey].shape):
                targetValues = torch.gather(outputs[self.outKey],1,targets.unsqueeze(-1))
            else:
                targetValues = torch.gather(outputs[self.outKey],1,targets)
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
        eigenSmooth: bool = False) -> torch.Tensor:

        weights = self.getCamWeights(inputDict,targetLayer,targets,activations,grads)
        weightedActivations:torch.Tensor = weights[:, :, None, None] * activations
        if eigenSmooth:
            cam = self.get2dProjection(weightedActivations)
        else:
            cam = weightedActivations.sum(axis=1)
        return cam
    def aggregateMultiLayers(self,camPerTargetLayer:torch.Tensor)->torch.Tensor:
        sizeList:list = [cam.size(-2)*cam.size(-1) for cam in camPerTargetLayer]
        maxIndex = sizeList.index(max(sizeList))
        maxSize = camPerTargetLayer[maxIndex].shape[-2:]
        camPerTargetLayer = torch.relu(torch.cat([F.interpolate(cam,size=maxSize,mode="bilinear") for cam in camPerTargetLayer], dim=1)) 
        result:torch.Tensor = camPerTargetLayer.mean(dim=1)
        result = (result-result.min()+1e-7)/(result.max()-result.min()+1e-7)
        return result

    def get2dProjection(self,activationBatch:torch.Tensor):
        activationBatch[torch.isnan(activationBatch)] = 0
        projections = []
        reshapedActivations = activationBatch.flatten(-2,-1).transpose(-1,-2)
        reshapedActivations = reshapedActivations - reshapedActivations.mean(dim=1)
        U, S, VT = torch.linalg.svd(reshapedActivations, full_matrices=True)
        projections = torch.matmul(reshapedActivations,VT[:,0].unsqueeze(-1)) 
        return projections.view(-1,*activationBatch.shape[2:])
    def __call__(
            self,
            inputDict:Dict[str,torch.Tensor],
            targets:Union[List[nn.Module],None]=None,
            augSmooth:bool = False,
            eigenSmooth:bool = False) -> torch.Tensor:
        if augSmooth is True:
            return None
        return self.forward(inputDict,targets,eigenSmooth)
    def __del__(self):
        self.activationsAndGrads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activationsAndGrads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
