import os
import sys
import dgl
sys.path.append(os.getcwd())
from typing import List,Dict
from scipy import sparse as sp
from dataloader.utils.text import getEncodeDict

from .loadTextGraphMethod import loadCustomTextGraph, loadTextGraph


def loadClsImageInfo(imagePath:str):
    returnData = []
    for root,_,fileList in os.walk(imagePath):
        try: 
            for file in fileList:
                returnData.append(os.path.join(root,file)) 
        except:
            pass
    returnData.sort()
    labelNameList = [os.path.basename(os.path.dirname(filePath)) for filePath in returnData]
    return {
        "imageFilePathList":returnData,
        "labelNameList":labelNameList
        }
def loadClsTextInfo(textPath:str):
    returnData = []
    for root,_,fileList in os.walk(textPath):
        try: 
            for file in fileList:
                returnData.append(os.path.join(root,file)) 
        except:
            pass
    returnData.sort()
    labelNameList = [os.path.basename(os.path.dirname(filePath)) for filePath in returnData]
    return {
        "textFilePathList":returnData,
        "labelNameList":labelNameList
        }
def loadClsCategoryList(path:str):
    categoryList = os.listdir(path)
    categoryList.sort()
    return {"categoryList":categoryList,"numClasses":len(categoryList)}
def loadTextEncodeDict(pathList:List[str]):
    fileAllPathList = []
    for path in pathList:
        for root,_,nameList in os.walk(path):
            fileAllPathList.extend([os.path.join(root,name).replace("\\","/") for name in nameList])
    fileAllPathList.sort()
    encodeDict = getEncodeDict(fileAllPathList)
    return {"encodeDict":encodeDict}

def loadKnowledgeGraphByRDF(pathList:List[str],outKey:str="",normalize:bool=True,undirected=True):
    def adjNormalize(adj:sp.coo_array):
        adjNew = adj + adj.transpose().multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adjNew:sp.coo_matrix = adjNew + sp.eye(adjNew.shape[0])
        return adjNew
    allPathList:List[str] = []
    nodeDict:Dict[str,int] = {}
    edgeDict:Dict[str,int] = {}
    U,V = [],[]
    for path in pathList:
        if os.path.isdir(path):
            for root,dirList,nameList in os.walk(allPathList):
                for name in nameList:
                    if os.path.splitext(name)[-1] == ".txt":
                        allPathList.append(os.path.join(root,name))
        elif os.path.isfile(path) and os.path.splitext(os.path.basename(path))[-1] == ".txt":
            allPathList.append(path)
    for path in allPathList:
        with open(path,"r") as file:
            for line in file.read().splitlines():
                h,r,t = line.split(" ")
                if h not in nodeDict.keys():
                    nodeDict.update({h:len(nodeDict.keys())})
                if r not in nodeDict.keys():
                    edgeDict.update({r:len(edgeDict.keys())})
                if t not in nodeDict.keys():
                    nodeDict.update({t:len(nodeDict.keys())})
                U.append(nodeDict[h])
                V.append(nodeDict[t])
    if undirected:
        U,V = U+V,V+U
    adj = sp.coo_matrix(([1 for i in range(len(U))],(U,V)),shape=[len(nodeDict.keys()),len(nodeDict.keys())])
    if normalize:
        adj = adjNormalize(adj)
    graph = dgl.from_scipy(adj)
    return {
        f"{outKey}NodeDict":nodeDict,
        f"{outKey}EdgeDict":edgeDict,
        f"{outKey}Adj":adj,
        f"{outKey}Graph":graph
    }