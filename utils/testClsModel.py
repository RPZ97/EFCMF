import argparse
import os
import sys

sys.path.append(os.getcwd())
from typing import List

import pydantic
from colorama import Fore, init

init(True)
class Args(pydantic.BaseModel):
    logPathList:List[str]
    localDevice:int
    deep:int
parser = argparse.ArgumentParser()
parser.add_argument("--logPathList",nargs="+",type=str)
parser.add_argument("--localDevice",default=-1,type=int)
parser.add_argument("--deep",default=6,type=int)
args = Args(**parser.parse_args().__dict__)
logPathList:List[str] = []
weightPathList:List[str] = []
for path in args.logPathList:
    pathDeep = len(path.replace("\\","/").split("/"))
    for root,_,fileNameList in os.walk(path):
        rootDeep = len(root.replace("\\","/").split("/"))
        deep = rootDeep-pathDeep
        if deep<=args.deep:
            for fileName in fileNameList:
                if fileName.split(".")[-1] == "pth":
                    logPathList.append(os.path.dirname(root))
                    weightPath = os.path.join(root,fileName)
                    weightPathList.append(weightPath)
                    print(Fore.GREEN+f"find weight Path:{weightPath}")
for logPath,weightPath in zip(logPathList,weightPathList):
    logPathDeep = len(logPath.replace("\\","/").split("/"))
    for root,_,fileNameList in os.walk(logPath):
        rootDeep = len(root.replace("\\","/").split("/"))
        deep = rootDeep-pathDeep
        if deep<=args.deep:
            for fileName in fileNameList:
                if fileName.split(".")[-1] == "yaml":
                    print(Fore.MAGENTA+os.path.join(root,fileName))
                    systemCommand = f"python main.py \
                        --mode test \
                        --setting \"{os.path.join(root,fileName)}\" \
                        --weight \"{weightPath}\" \
                        --logPath \"{root}\" \
                        --localDevice {args.localDevice} \
                        "
                    os.system(systemCommand)