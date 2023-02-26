import os
import sys

sys.path.append(os.getcwd())

from PIL import Image
from torch.utils.data import Dataset

from dataloader.utils.image import getImageTransform


class ImageDataset(Dataset):
    def __init__(
        self,
        imageSize=[448,448],
        cropSize=[384,384],
        random=True,
        keyName:str="imageFilePathList",
        *args, **kwargs) -> None:

        super(ImageDataset,self).__init__()
        self.imageTransform = getImageTransform(imageSize=imageSize,cropSize=cropSize,random=random)
        for key,value in kwargs.items():
            if key == keyName:
                self.imageFilePathList = value
    def __getitem__(self, index):
        imagePIL = Image.open(self.imageFilePathList[index]).convert("RGB")
        imageTensor = self.imageTransform(imagePIL) 
        return {"image":imageTensor}
    def __len__(self):
        return len(self.imageFilePathList)