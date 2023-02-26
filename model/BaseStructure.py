from torch import nn


class BaseStructure(nn.Module):
    def __init__(self,parameter,datasetInfo) -> None:
        super(BaseStructure,self).__init__()
        self.modelParameter = parameter
        self.datasetInfo = datasetInfo