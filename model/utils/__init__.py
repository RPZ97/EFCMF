from typing import Dict, List, Union

from torch import nn


def getHookOut(out:dict,name):
    def hook(model:nn.Module,input,output):
        out[name] = output
    return hook

def createGraphModel(model:nn.Module,nodes:Union[Dict[str,str],List[str]]):
    class GraphModel(nn.Module):
        def __init__(self) -> None:
            super(GraphModel,self).__init__()
            self.registerModel = model
            self.outs = {}
            if type(nodes) == type({}):
                nodeIter:List[str] = nodes.keys()
            elif type(nodes) == type([]):
                nodeIter:List[str] = nodes
            for nodeGetName in nodeIter:
                nodeGetName:str
                nodeGetNameList = nodeGetName.split(".")
                layer = self.registerModel
                for nodeRegisterName in nodeGetNameList:
                    layer:nn.Module = getattr(layer,nodeRegisterName)
                layer.register_forward_hook(getHookOut(self.outs,nodeGetName))
        def forward(self,x):
            model(x)
            if type(nodes) == type({}):
                return {nodeOutName:self.outs[nodeGetName] for nodeGetName,nodeOutName in nodes.items()}
            elif type(nodes) == type([]):
                return [self.outs[node] for node in nodes]
    return GraphModel()