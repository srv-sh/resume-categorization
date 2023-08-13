# resume-categorization

## Installation

```
conda create -n rc python==3.9.0
conda activate rc

```
After activate environment ```rc``` just run ```pip install -r requirements.txt```

## Datasets
I utillized two kaggle dataset to train the model
1. https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset (given)
2. https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset (additional)

also i augmented few dataset to overcome overfitting issue

 in this url can be found the overall dataset which i used to train the model. 

 ## Files

| File Name        | Description |
| -----------      | ----------- |
| eda.ipynb        | This file contains preprocessing steps, data augmentation, and exploratory data analysis|
| model.ipynb      | This file contrains code of model building and training steps        |
| script.py        | This file takes two parameters. one is ckpt file path and another one is data directory file path        |


 ## Inferance

 Firstly download the checkpoint from this [LINK] and unzip it to the root directory.

 then run below script 

 ``` python script.py --path path/to/dir --ckpt path/to/ckpt ```
 



