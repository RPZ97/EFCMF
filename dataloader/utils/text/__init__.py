import threading

import torch
from torch import nn
from tqdm import tqdm
from typing import List,Dict

def getEncodeDict(pathList:list):
    encodeDict = {}
    for path in pathList:
        with open(path,"r") as file:
            wordList = " ".join(file.read().splitlines()).split(" ")
            for word in wordList:
                if word not in encodeDict.keys():
                    encodeDict.update({word:len(encodeDict.keys())})
    return encodeDict

def getEmbeddingWeight(embeddingPath,encodeDict:Dict[str,int],threadNums=1000,method="xavier_uniform",initArgs={})->torch.FloatTensor:
    with open(embeddingPath,"r",encoding="utf-8") as embeddingFile:
        embeddingData = embeddingFile.read().splitlines()
        embeddingTensor = torch.zeros(len(encodeDict.keys()),len(embeddingData[0].split(" "))-1)
        getattr(nn.init,method)(embeddingTensor,**initArgs)
        def setEmbeddingTensor(embeddingData,encodeDict:Dict,start,end):
            for data in embeddingData[start:end]:
                dataSplit = data.split(" ")
                word = dataSplit[0]
                if word in encodeDict.keys():
                    tensor = torch.FloatTensor([float(i) for i in dataSplit[1:]])
                    try:
                        embeddingTensor[encodeDict[word]] = tensor
                    except:
                        pass
        splitNums = int(len(embeddingData)/threadNums)+1
        threadList:List[threading.Thread] = []
        for i in range(splitNums):
            thread = threading.Thread(target=setEmbeddingTensor,args=(embeddingData,encodeDict,i*threadNums,(i+1)*threadNums))
            threadList.append(thread) 
            thread.start()
        for thread in threadList:
            thread.join()
        return embeddingTensor