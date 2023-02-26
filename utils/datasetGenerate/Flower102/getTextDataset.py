import argparse
import os
import re
from typing import Dict, List

import pandas as pd
import texthero
from tqdm import tqdm

basePath = "data/Flower102"
savePath = "dataprocess/Flower102"

parser = argparse.ArgumentParser()

parser.add_argument("--text",default=os.path.join(basePath,"text_c10"))
parser.add_argument("--textSave",default="dataprocess/Flower102/text")
parser.add_argument("--textClean",default="dataprocess/Flower102/textClean")
parser.add_argument("--labelFile",default=os.path.join(basePath,"imagelabels.mat"))
parser.add_argument("--splitFile",default=os.path.join(basePath,"setid.mat"))

args = parser.parse_args()

def setSplit(fileDirectory,saveDirectory,labelList:List[int],setSplitDict:Dict[str,List[int]],suffixList:List[str]=["txt"]):
    filePathList:List[str] = []
    for root,_,nameList in os.walk(fileDirectory):
        for name in nameList:
            suffix = name.split(".")[-1]
            if suffix in suffixList:
                if os.path.isfile(os.path.join(root,name)):
                    filePathList.append(os.path.join(root,name))
    filePathList.sort(key=lambda x:int(re.findall('\d+',os.path.basename(x))[0]))
    setDict:Dict[int,str] = {}
    for setName,idList in setSplitDict.items():
        for id in idList:
            setDict.update({int(id):setName})
    iteration = tqdm(filePathList)
    for i,filePath in enumerate(iteration):
        with open(filePath,"r") as file:
            data = file.read().splitlines()
            df = pd.Series(data)
            dfClean = texthero.clean(df)
        fileName = os.path.basename(filePath)
        labelId = int(labelList[i])
        fileId = int(re.findall('\d+',fileName)[0])
        targetPath = os.path.join(saveDirectory,setDict[fileId],str(labelId))
        if not os.path.isdir(targetPath):
            os.makedirs(targetPath)
        saveFilePath = os.path.join(targetPath,fileName)
        with open(saveFilePath,"w") as saveFile:
            saveFile.write("\n".join(dfClean.to_list()))
        iteration.set_description(f"{filePath} -> {saveFilePath}")
        
if __name__ == "__main__":
    from scipy.io import loadmat
    setSplitDictOrigin = loadmat(args.splitFile)
    setSplitDict = {
        "train":setSplitDictOrigin["trnid"][0],
        "val":setSplitDictOrigin["valid"][0],
        "test":setSplitDictOrigin["tstid"][0],
    }
    labelList = loadmat(args.labelFile)["labels"][0]
    setSplit(args.text,args.textClean,labelList,setSplitDict)