import argparse
import os
import sys

sys.path.append(os.getcwd())
from typing import List

import numpy as np
import pandas as pd
import pydantic
from colorama import Fore, init
from matplotlib import pyplot as plt

init(True)
class Args(pydantic.BaseModel):
    excelPathList:List[str]
    savePath:str
    deep:int
parser = argparse.ArgumentParser()
parser.add_argument("--excelPathList",default=[],nargs="+",type=str)
parser.add_argument("--savePath",default="",type=str)
parser.add_argument("--deep",default=0,type=int)
args = Args(**parser.parse_args().__dict__)

def plotHeatmap(dataFrame:pd.DataFrame,savePath:str,scaleMin:float=0,scaleMax:float=1):
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 全局字体，显示中文，宋体，可替换为其他字体
    # plt.rcParams['font.size'] = 14 # 全局字号
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig, ax = plt.subplots(dpi=1200)
    indexList:List[str] = list(dataFrame.index)
    columnList:List[str] = list(dataFrame.columns)
    matrix = dataFrame.to_numpy()
    im = ax.imshow((matrix-scaleMin)/(scaleMax-scaleMin),cmap="jet",vmin=scaleMin,vmax=scaleMax)
    ax.set_xticklabels([""]+columnList)
    ax.set_yticklabels([""]+indexList)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    for i,category in enumerate(indexList):
        for j,category in enumerate(columnList):
            text = ax.text(j,i,f"{matrix[i,j]:.4f}",ha="center",va="center",color="w")
    fig.savefig(savePath,bbox_inches='tight')
def main():
    if not os.path.isdir(args.savePath):
        os.makedirs(args.savePath)
    excelPathList = []
    for excelPath in args.excelPathList:
        if os.path.isdir(excelPath):
            excelPathDeep = len(excelPath.replace("\\","/").split("/"))
            for root,_,nameList in os.walk(excelPath):
                rootDeep = len(root.replace("\\","/").split("/"))
                if rootDeep-excelPathDeep == args.deep:
                    for name in nameList:
                        if os.path.splitext(name)[1] in [".csv"] :
                            excelPathList.append(os.path.join(root,name))
                            print(Fore.BLUE+f"find file:{os.path.join(root,name)}") 
        elif os.path.isfile(excelPath):
            if os.path.splitext(excelPath)[1] in [".csv"] :
                excelPathList.append(excelPath)
                print(Fore.BLUE+f"find file:{excelPath}") 
    if not(os.path.isdir(args.savePath)):
        os.makedirs(args.savePath)
    for k,excelPath in enumerate(excelPathList):
        dataFrame:pd.DataFrame = pd.read_csv(excelPath,header=0,index_col=0)
        dataFrame.fillna(0,inplace=True)
        imageFilePath = os.path.join(args.savePath,f"{os.path.splitext(os.path.basename(excelPath))[0]}.png")
        plotHeatmap(dataFrame,imageFilePath)
        print(Fore.GREEN+f"write:{imageFilePath}")
main()