from PIL import Image
from torchvision import transforms


def getImageTransform(imageSize=[448,448],cropSize=[448,448],random=False):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    if random:
        imageTransform = transforms.Compose([
                transforms.Resize(imageSize,Image.BILINEAR),  # Let smaller edge match
                transforms.RandomCrop(cropSize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize(norm_mean, norm_std),
            ])
    else:
        imageTransform = transforms.Compose([
                transforms.Resize(imageSize,Image.BILINEAR),  # Let smaller edge match
                transforms.CenterCrop(cropSize),
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize(norm_mean, norm_std),
            ])
    return imageTransform