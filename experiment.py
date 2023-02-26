import argparse
import os
from typing import Any, Dict, List, Union

import pydantic
import yaml
from colorama import Fore, init

init(True)
class Command(pydantic.BaseModel):
    program:str = "python"
    name:str 
    parameter:Dict[str,Union[list,str]]

class Args(pydantic.BaseModel):
    experimentList:List[str]
parser = argparse.ArgumentParser()
parser.add_argument("-e","--experimentList",default=[
    "setting/experiment"
    ],nargs="+",type=str)
args = Args(**parser.parse_args().__dict__)
def getParameterString(parameter:Dict[str,Any]):
    parameterString = ""
    for key,value in parameter.items():
        valueString = ""
        if type(value) == list:
            valueString = " ".join([str(i) for i in value])
        else:
            valueString = f"{value}"
        parameterString+= f" --{key} {valueString}"
    return parameterString
def main():
    filePathList = []
    for experiment in args.experimentList:
        if os.path.isdir(experiment):
            for root,_,nameList in os.walk(experiment):
                for name in nameList:
                    if(os.path.splitext(name)[1]) == ".yaml":
                        filePathList.append(os.path.join(root,name))
        elif os.path.isfile(experiment):
            if(os.path.splitext(experiment)[1]) == ".yaml":
                filePathList.append(experiment)
    for filePath in filePathList:
        with open(filePath,"r") as filePath:
            fileData = yaml.safe_load(filePath)
            for commandData in fileData:
                experimendCommand = Command(**commandData) 
                result = os.system(
                    
                    f"\
                    {experimendCommand.program} \
                    {experimendCommand.name} \
                    {getParameterString(experimendCommand.parameter)}"
                    )  
                if result == 0:
                    print(Fore.GREEN+f"The command executes smoothly: {filePath}")
                else:
                    print(Fore.RED+f"error: {filePath}")
                    break
if __name__ == "__main__":
    main()