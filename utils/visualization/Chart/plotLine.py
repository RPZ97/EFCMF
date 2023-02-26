import argparse
import os
import sys

sys.path.append(os.getcwd())
from typing import List

import pandas as pd
import pydantic
from colorama import Fore, init
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator,MultipleLocator,FormatStrFormatter
import numpy as np
init(True)
class Args(pydantic.BaseModel):
    excelPathList:List[str]
    savePath:str
    deep:int
    meanName:str
    medianName:str
    minName:str
    maxName:str
    plotIndex:List[int]
parser = argparse.ArgumentParser()
parser.add_argument("--excelPathList",default=[""],nargs="+",type=str)
parser.add_argument("--savePath",default="",type=str)
parser.add_argument("--deep",default=0,type=int)
parser.add_argument("--meanName",default="Mean",type=str)
parser.add_argument("--medianName",default="Median",type=str)
parser.add_argument("--maxName",default="Max",type=str)
parser.add_argument("--minName",default="Min",type=str)
parser.add_argument("--plotIndex",default=[],nargs="+",type=int)
args = Args(**parser.parse_args().__dict__)

linemarkerstyles=['.','o','v','s','^','p','*','+','<','>','x','|','_']
def main():
    if not os.path.isdir(args.savePath):
        os.makedirs(args.savePath)
    excelPathList = []
    for excelPath in args.excelPathList:
        baseDeep = len(excelPath.replace("\\","/").split("/"))
        if os.path.isdir(excelPath):
            for root,_,nameList in os.walk(excelPath):
                rootDeep = len(root.replace("\\","/").split("/"))
                deep = rootDeep-baseDeep
                if deep == args.deep:
                    for name in nameList:
                        if os.path.splitext(name)[1] in [".csv"] :
                            excelPathList.append(os.path.join(root,name))
                            print(Fore.BLUE+f"find file:{os.path.join(root,name)}") 
        elif os.path.isfile(excelPath):
            if os.path.splitext(excelPath)[1] in [".csv"] :
                excelPathList.append(excelPath)
                print(Fore.BLUE+f"find file:{excelPath}") 
    for excelPath in excelPathList:

        dataFrame:pd.DataFrame = pd.read_csv(excelPath,header=0,index_col=0)
        dataFrame.fillna(0,inplace=True)
        # if args.meanName!="":
        #     dataFrame.loc[args.meanName] = dataFrame.to_numpy().mean(axis=0)
        # if args.medianName!="":
        #     dataFrame.loc[args.medianName] = np.median(dataFrame.to_numpy(),axis=0)
        # if args.minName!="":
        #     dataFrame.loc[args.minName] = dataFrame.to_numpy().min(axis=0)
        # if args.maxName!="":
        #     dataFrame.loc[args.maxName] = dataFrame.to_numpy().max(axis=0)
        dataNumpy = dataFrame.to_numpy()
        plt.clf()
        # plt.subplots_adjust(right=0.)
        plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 全局字体，显示中文，宋体，可替换为其他字体
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
        # with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots(figsize=(6,3),dpi=600)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2/5))
        # ax.grid(ls=':',lw=0.8,color='grey')
        ax.set_ylim(0,1)
        columnList = list(dataFrame.columns)
        indexList = np.array(dataFrame.index)
        if len(args.plotIndex)>0:
            for i,index in enumerate(args.plotIndex):
                ax.plot(
                    columnList,dataNumpy[index],
                    label=indexList[index],
                    marker=linemarkerstyles[i%len(linemarkerstyles)],ls='--',markersize=3)
            ax.legend(list(indexList[args.plotIndex]), loc='upper left',bbox_to_anchor=[1.05,1],frameon=False)
        else:
            for i,index in enumerate(indexList):
                ax.plot(
                    columnList,dataNumpy[i],
                    label=index,
                    marker=linemarkerstyles[i%len(linemarkerstyles)],ls='--',markersize=3)
            ax.legend(list(indexList), loc='upper left',bbox_to_anchor=[1,1],frameon=False)
        saveFilePath =os.path.join(args.savePath,os.path.splitext(os.path.basename(excelPath))[0]+".png")
        fig.tight_layout()

        fig.savefig(saveFilePath,dpi=600, bbox_inches='tight')
        print(Fore.GREEN+f"save fig:{saveFilePath}") 
main()