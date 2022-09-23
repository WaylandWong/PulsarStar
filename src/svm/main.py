import os
import sys

import numpy as np
import configparser
import pandas as pd
import matplotlib.pyplot as plt  # for data visualization
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from pathlib import Path

# 定义默认配置
ROOT_DIR = Path(__file__).parent.parent.parent
data_path = ROOT_DIR / "data"
warnings.filterwarnings('ignore')
train_data = data_path / "HTRU_2.csv"
cp = configparser.ConfigParser()

# 训练集 标签 测试集 标签
X_train = None
y_train = None
X_test = None
y_test = None


# 加载配置文件
def load_config():
    global cp
    cp.read(ROOT_DIR / "config/svm.cfg")


# 用图像的形式展示数据集基本统计信息
def show_data_set_summary(df: pd.DataFrame):
    sta = round(df.describe(), 2)
    print(sta)

    plt.figure()

    plt.subplot(4, 2, 1)
    fig = df.boxplot(column='IP Mean')
    fig.set_title('')
    fig.set_ylabel('IP Mean')

    plt.subplot(4, 2, 2)
    fig = df.boxplot(column='IP Sd')
    fig.set_title('')
    fig.set_ylabel('IP Sd')

    plt.subplot(4, 2, 3)
    fig = df.boxplot(column='IP Kurtosis')
    fig.set_title('')
    fig.set_ylabel('IP Kurtosis')

    plt.subplot(4, 2, 4)
    fig = df.boxplot(column='IP Skewness')
    fig.set_title('')
    fig.set_ylabel('IP Skewness')

    plt.subplot(4, 2, 5)
    fig = df.boxplot(column='DM-SNR Mean')
    fig.set_title('')
    fig.set_ylabel('DM-SNR Mean')

    plt.subplot(4, 2, 6)
    fig = df.boxplot(column='DM-SNR Sd')
    fig.set_title('')
    fig.set_ylabel('DM-SNR Sd')

    plt.subplot(4, 2, 7)
    fig = df.boxplot(column='DM-SNR Kurtosis')
    fig.set_title('')
    fig.set_ylabel('DM-SNR Kurtosis')

    plt.subplot(4, 2, 8)
    fig = df.boxplot(column='DM-SNR Skewness')
    fig.set_title('')
    fig.set_ylabel('DM-SNR Skewness')


# 用图像的形式显示数据的分布情况
def show_data_set_distribution(df: pd.DataFrame):
    plt.figure()

    plt.subplot(4, 2, 1)
    fig = df['IP Mean'].hist(bins=20)
    fig.set_xlabel('IP Mean')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 2)
    fig = df['IP Sd'].hist(bins=20)
    fig.set_xlabel('IP Sd')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 3)
    fig = df['IP Kurtosis'].hist(bins=20)
    fig.set_xlabel('IP Kurtosis')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 4)
    fig = df['IP Skewness'].hist(bins=20)
    fig.set_xlabel('IP Skewness')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 5)
    fig = df['DM-SNR Mean'].hist(bins=20)
    fig.set_xlabel('DM-SNR Mean')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 6)
    fig = df['DM-SNR Sd'].hist(bins=20)
    fig.set_xlabel('DM-SNR Sd')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 7)
    fig = df['DM-SNR Kurtosis'].hist(bins=20)
    fig.set_xlabel('DM-SNR Kurtosis')
    fig.set_ylabel('Number of pulsar stars')

    plt.subplot(4, 2, 8)
    fig = df['DM-SNR Skewness'].hist(bins=20)
    fig.set_xlabel('DM-SNR Skewness')
    fig.set_ylabel('Number of pulsar stars')


def show_data_set_view(df: pd.DataFrame):
    show_view = cp.getboolean('Dataset', 'SHOW_VIEW')
    if show_view:
        show_data_set_summary(df)
        show_data_set_distribution(df)

        # plt.show()    # 直接阻塞显示窗口
        plt.pause(10)  # 等待10s后关闭所有显示窗口
        plt.close('all')


# 数据预处理
def date_preprocess(df: pd.DataFrame):
    # 区分特征向量和目标值
    X = df.drop(['target_class'], axis=1)
    y = df.get('target_class')

    # 分割训练集为训练集(80%)和测试集(20%)
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    # print(X_train.shape)
    # print(X_test.shape)

    # 特征缩放
    cols = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    # print(X_train.shape)
    # print(y_train.shape)


# svm
# 使用默认超参数训练并进行预测
# C=1.0, kernel=rbf, gamma=auto
def svm_train(kernel="rbf", C=1.0, gamma='auto'):
    global X_train, y_train, X_test, y_test
    svc = SVC(kernel=kernel, C=C, gamma=gamma)

    svc.fit(X_train, y_train)

    # 预测
    y_pred = svc.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    y_pred_train = svc.predict(X_train)
    train_score = accuracy_score(y_train, y_pred_train)

    # print(score)
    print('Train Model accuracy score with kernel:{0}, C:{1}, gamma:{2} hyper parameters: {3:0.4f}'
          .format(kernel, C, gamma, train_score))
    print('Test Model accuracy score with kernel:{0}, C:{1}, gamma:{2} hyper parameters: {3:0.4f}'
          .format(kernel, C, gamma, test_score))


def check_test_distribution():
    counts = y_test.value_counts()
    null_accuracy = (counts[0] / (counts[0] + counts[1]))

    print('Null accuracy score: {0:0.4f}'.format(null_accuracy))


# 线性
def svm_linear():
    print("## linear kernel ##")

    kernel = 'linear'
    svm_train(kernel=kernel)

    # linear kernel with C=1.0
    svm_train(kernel=kernel, C=1.0)

    # linear kernel with C=100.0
    svm_train(kernel=kernel, C=100.0)

    # linear kernel with C=1000.0
    svm_train(kernel=kernel, C=1000.0)


def svm_polynomial():
    print("## polynomial kernel ##")


def main():
    load_config()

    # 读取训练样本文件
    df = pd.read_csv(train_data)
    # 脉冲轮廓宽度的平均值 脉冲轮廓宽度的标准差 脉冲轮廓宽度高于阈值的脉冲占比  脉冲轮廓宽度偏度
    # DM-SNR(色散度-信噪比)曲线平均值 DM-SNR曲线标准差 DM-SNR曲线高于阈值的脉冲占比 DM-SNR曲线偏度 目标类型(label)
    df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness',
                  'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']

    show_data_set_view(df)
    date_preprocess(df)

    svm_linear()

    check_test_distribution()


# 程序入口
if __name__ == "__main__":
    main()
