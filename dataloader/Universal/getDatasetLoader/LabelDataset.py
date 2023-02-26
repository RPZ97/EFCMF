from typing import List

import torch
from torch.utils.data import Dataset


class LabelDataset(Dataset):
    def __init__(
        self,
        categoryList:List[str],
        keyName:str="labelNameList",
        *args, **kwargs:dict) -> None:
        self.labelList = []
        super(LabelDataset,self).__init__()
        self.categoryDict = {category:i for i,category in enumerate(categoryList)}
        for key,value in kwargs.items():
            if key == keyName:
                self.labelList = [self.categoryDict[labelName] for labelName in value]
    def __getitem__(self, index):
        labelTensor = torch.LongTensor([self.labelList[index]]).view(-1)[0]
        return {"label":labelTensor}
    def __len__(self):
        return len(self.labelList)