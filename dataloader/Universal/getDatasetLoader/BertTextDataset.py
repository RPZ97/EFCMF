from typing import List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BertTextDataset(Dataset):
    def __init__(
        self,
        textFilePathList:List[str],
        pretrained="bert-base-cased",
        *args, **kwargs
        ) -> None:
        super(BertTextDataset,self).__init__()
        self.textFilePathList = textFilePathList
        self.tokenizer:BertTokenizer = BertTokenizer.from_pretrained(pretrained)
        # self.encodeDict = {self.tokenizer.convert_ids_to_tokens(i):i  for i in range(self.tokenizer.vocab_size)}
    def __getitem__(self, index):
        textPath = self.textFilePathList[index]
        with open(textPath,"r") as textFile:
            textData = textFile.read().splitlines()
            tokenData = self.tokenizer(textData)
            inputIds,attentionMask = tokenData["input_ids"],tokenData["attention_mask"]
            encodeList = sum(inputIds,[])
            maskList = sum(attentionMask,[])
            return {
                "text":torch.LongTensor(encodeList),
                "mask":torch.LongTensor(maskList)} 
    def __len__(self):
        return len(self.textFilePathList)