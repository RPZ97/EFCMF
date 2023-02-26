import argparse
import test

import yaml

import train
from Interface import Args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="train")
    parser.add_argument("--cpu",action='store_true',default=False)
    parser.add_argument("--setting",default="setting/CUB2002011/EFCMF/EFCMF_SwinTransformer-TextCNN.yaml")
    parser.add_argument("--logPath",default="")
    parser.add_argument("--weight",default="")
    parser.add_argument("--distributed",action='store_true',default=False)
    parser.add_argument("--distBackend",default="nccl",type=str)
    parser.add_argument('--rank', default=0,type=int,help='rank of current process')
    parser.add_argument('--localDevice', default=-1,type=int,help='local device')
    parser.add_argument('--worldSize', default=1,type=int,help="world size")
    parser.add_argument('--initMethod', default='tcp://127.0.0.1:9005',help="init method")
    parser.add_argument("--resetKeyList",default=[],nargs='+')
    parser.add_argument("--resetValueList",default=[],nargs="+")
    args = Args(**parser.parse_args().__dict__)
    setting = yaml.safe_load(open(getattr(args,"setting"),'r',encoding='utf-8')) 
    assert len(args.resetKeyList) == len(args.resetValueList)
    if args.mode == "train":
        trainMethod:train.BaseTrain = getattr(train,setting["trainSetting"]["method"])(args,setting)
        trainMethod.train()
    elif args.mode == "test":
        testMethod:test.BaseTest = getattr(test,setting["testSetting"]["method"])(args,setting)
        testMethod.test()
if __name__ == "__main__":
    main()