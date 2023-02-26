from typing import Dict, List

import torch


def textCollateFn(dataDict:Dict[str,List[torch.Tensor]]):
    device = dataDict["text"][0].device
    lengthList = [
        len(textData)
        for textData in dataDict["text"]
    ]
    maxLength = max(lengthList)
    tensorList = [
        torch.cat([textData,torch.zeros(maxLength-lengthList[i],device=textData.device,dtype=textData.dtype)])
        for i,textData in enumerate(dataDict["text"]) 
        ]
    maskList = [
        torch.cat([maskData,torch.zeros(maxLength-lengthList[i],device=device)])
        for i,maskData in enumerate(dataDict["mask"])]
    returnDict = {
        "text":torch.stack(tensorList),
        "mask":torch.stack(maskList),
        "length":torch.LongTensor(lengthList)
        }
    return returnDict   

def imageCollateFn(dataDict:Dict[str,List[torch.Tensor]]):
    return {
        "image":torch.stack(dataDict["image"])
    }
def labelCollateFn(dataDict:Dict[str,List[torch.Tensor]]):
    return {
        "label":torch.stack(dataDict["label"])
    }
def indexCollateFn(dataDict:Dict[str,List[torch.Tensor]]):
    return {
        "index":torch.stack(dataDict["index"])
    }
def pathCollateFn(dataDict:Dict[str,List[torch.Tensor]]):
    return {
        "path":dataDict["path"]
    }