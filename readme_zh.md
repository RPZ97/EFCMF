# 介绍

这里是论文[&#34;EFCMF: A Multimodal Robustness Enhancement Framework for Fine-Grained Recognition&#34;](https://www.mdpi.com/2076-3417/13/3/1640)的代码

如果论文代码能够对您的研究有所帮助，
您可以引用本文：

```
@Article{app13031640,
AUTHOR = {Zou, Rongping and Zhu, Bin and Chen, Yi and Xie, Bo and Shao, Bin},
TITLE = {EFCMF: A Multimodal Robustness Enhancement Framework for Fine-Grained Recognition},
JOURNAL = {Applied Sciences},
VOLUME = {13},
YEAR = {2023},
NUMBER = {3},
ARTICLE-NUMBER = {1640},
URL = {https://www.mdpi.com/2076-3417/13/3/1640},
ISSN = {2076-3417},
ABSTRACT = {Fine-grained recognition has many applications in many fields and aims to identify targets from subcategories. This is a highly challenging task due to the minor differences between subcategories. Both modal missing and adversarial sample attacks are easily encountered in fine-grained recognition tasks based on multimodal data. These situations can easily lead to the model needing to be fixed. An Enhanced Framework for the Complementarity of Multimodal Features (EFCMF) is proposed in this study to solve this problem. The model’s learning of multimodal data complementarity is enhanced by randomly deactivating modal features in the constructed multimodal fine-grained recognition model. The results show that the model gains the ability to handle modal missing without additional training of the model and can achieve 91.14% and 99.31% accuracy on Birds and Flowers datasets. The average accuracy of EFCMF on the two datasets is 52.85%, which is 27.13% higher than that of Bi-modal PMA when facing four adversarial example attacks, namely FGSM, BIM, PGD and C&W. In the face of missing modal cases, the average accuracy of EFCMF is 76.33% on both datasets respectively, which is 32.63% higher than that of Bi-modal PMA. Compared with existing methods, EFCMF is robust in the face of modal missing and adversarial example attacks in multimodal fine-grained recognition tasks. The source code is available at https://github.com/RPZ97/EFCMF (accessed on 8 January 2023).},
DOI = {10.3390/app13031640}
}
```

# 运行步骤

## 安装环境

安装requirement.txt中所需要的包

然后运行命令

```
python experiment.py
```

```

```

## 代码结构

* data

  * 存放原始数据
* dataloader

  * 数据集生成器以及数据载入器
* experiments

  * 实验运行方式yaml文件保存地址
* loss

  * 损失函数
* model

  * 论文模型
* setting

  * 模型和训练的参数设置
* test

  * 推理过程代码
* train

  * 模型训练过程
* utils

  * 数据集划分工具

## 链接数据集

本文采用的两个数据集CUB-200-2011以及Flowers102，需要将数据集放置在data文件夹下面

数据集结构如下

* data
  * Flower102
    * imageslabels.mat
    * setid.mat
    * images
  * CUB2002011

## 数据预处理

utils/datasetGenerate为数据预处理代码

## 模型训练

运行 ``python main.py --setting`` 配置文件 即可训练模型

## 模型推理

运行 ``python main.py --setting 日志中的配置文件 --weight 日志中的权重 ``即可进行推理
