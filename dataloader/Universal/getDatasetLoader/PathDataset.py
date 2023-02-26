from torch.utils.data import Dataset

class PathDataset(Dataset):
    def __init__(
        self,
        keyName:str="imageFilePathList",
        *args, **kwargs) -> None:

        super(PathDataset,self).__init__()
        for key,value in kwargs.items():
            if key == keyName:
                self.imageFilePathList = value
    def __getitem__(self, index):
        return {"path":self.imageFilePathList[index]}
    def __len__(self):
        return len(self.imageFilePathList)