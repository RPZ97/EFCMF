import os
import sys

sys.path.append(os.getcwd())

from typing import List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self,textFilePathList:List[str],encodeDict:dict, *args, **kwargs) -> None:
        super(TextDataset,self).__init__()
        self.encodeDict = encodeDict
        self.textFilePathList = textFilePathList
    def __getitem__(self, index):
        textPath = self.textFilePathList[index]
        with open(textPath,"r") as textFile:
            textData =" ".join(textFile.read().splitlines()).split(" ")
            encodeList = [self.encodeDict[word] for word in textData]
            maskList = [1 for i in range(len(encodeList))]
            return {
                "text":torch.LongTensor(encodeList),
                "mask":torch.LongTensor(maskList)} 
    def __len__(self):
        return len(self.textFilePathList)