import torch
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(
        self,
        keyName:str,
        *args, **kwargs) -> None:

        super(IndexDataset,self).__init__()
        self.keyName = keyName
        self.indexList = []
        for key,value in kwargs.items():
            if key == keyName:
                self.indexList = value
    def __getitem__(self, index):
        indexTensor = torch.LongTensor([self.indexList[index]]) 
        return {"index":indexTensor}
    def __len__(self):
        return len(self.indexList)