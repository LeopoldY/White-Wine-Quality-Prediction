# python dataloader.py

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

def load_from_csv(file_path:str, split_rate:float):

    X, y = get_x_y(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = split_rate, random_state=1
    ) # 划分训练/测试数据

    # 标准化数据集：
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 将ndarray转化为tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long().add(-3) # (0 ~ 6)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long().add(-3) # (0 ~ 6)

    # 创建张量数据集 TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

def get_x_y(file_path:str):
    data = pd.read_csv(file_path, delimiter=';') # 载入数据集
    X = data.iloc[:, :-1].values # 分离出参数 X
    y = data.iloc[:, -1].values # 分离出标签 Y (3 ~ 9)
    return X, y