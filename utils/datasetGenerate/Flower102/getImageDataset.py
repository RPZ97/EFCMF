import argparse
import os
import re
import shutil
from typing import Dict, List

from tqdm import tqdm

basePath = "data/Flower102"
savePath = "dataprocess/Flower102"

parser = argparse.ArgumentParser()
parser.add_argument("--image",default=os.path.join(basePath,"images"))
parser.add_argument("--imageSave",default=os.path.join(savePath,"images"))
parser.add_argument("--labelFile",default=os.path.join(basePath,"imagelabels.mat"))
parser.add_argument("--splitFile",default=os.path.join(basePath,"setid.mat"))


args = parser.parse_args()

def setSplit(fileDirectory,saveDirectory,labelList:List[int],setSplitDict:Dict[str,List[int]]):
    fileNameList = [fileName for i,fileName in enumerate(os.listdir(fileDirectory)) if os.path.isfile(os.path.join(fileDirectory,fileName))]
    fileNameList.sort(key=lambda x:int(re.findall('\d+',x)[0]))
    setDict:Dict[int,str] = {}
    for setName,idList in setSplitDict.items():
        for id in idList:
            setDict.update({int(id):setName})
    iteration = tqdm(fileNameList)
    for i,fileName in enumerate(iteration):
        filePath = os.path.join(fileDirectory,fileName)
        labelId = int(labelList[i])
        fileId = int(re.findall('\d+',fileName)[0])
        targetPath = os.path.join(saveDirectory,setDict[fileId],str(labelId))
        if not os.path.isdir(targetPath):
            os.makedirs(targetPath)
        copyFilePath = os.path.join(targetPath,fileName)
        iteration.set_description(f"{filePath} -> {copyFilePath}")
        if not os.path.isfile(copyFilePath):
            shutil.copy(filePath,copyFilePath)
if __name__ == "__main__":
    from scipy.io import loadmat
    setSplitDictOrigin = loadmat(args.splitFile)
    setSplitDict = {
        "train":setSplitDictOrigin["trnid"][0],
        "val":setSplitDictOrigin["valid"][0],
        "test":setSplitDictOrigin["tstid"][0],
    }
    print(setSplitDict["train"])
    labelList = loadmat(args.labelFile)["labels"][0]
    setSplit(args.image,args.imageSave,labelList,setSplitDict)