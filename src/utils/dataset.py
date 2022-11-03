import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import configparser
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
cp = configparser.ConfigParser()
# 加载配置文件
cp.read(ROOT_DIR / "config/svm.cfg")
# 定义默认配置
ROOT_DIR = Path(__file__).parent.parent.parent
data_path = ROOT_DIR / "data"
train_data = data_path / "HTRU_2.csv"


# 数据预处理
def data_preprocess():
    # 读取训练样本文件
    df = pd.read_csv(train_data)

    # 脉冲轮廓宽度的平均值 脉冲轮廓宽度的标准差 脉冲轮廓宽度高于阈值的脉冲占比  脉冲轮廓宽度偏度
    # DM-SNR(色散度-信噪比)曲线平均值 DM-SNR曲线标准差 DM-SNR曲线高于阈值的脉冲占比 DM-SNR曲线偏度 目标类型(label)
    df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness',
                  'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
    show_data_set_view(df)
    # 区分特征向量和目标值
    y = df.get('target_class')

    # 分割训练集为训练集(80%)和测试集(20%)
    x_train, x_test, y_train, y_test = \
        train_test_split(df, y, test_size=0.2, random_state=3)

    # 下采样
    # train_1 = x_train[x_train['target_class'] == 1]
    # train_0 = x_train[x_train['target_class'] == 0]
    # sample = train_0.sample(train_1.shape[0])
    # train_all = train_1.append(sample)

    # 构造训练集 测试集
    # svm.y_train = train_all.get('target_class')
    # svm.x_train = train_all.drop(['target_class'], axis=1)

    y_train = x_train.get('target_class')
    x_train = x_train.drop(['target_class'], axis=1)

    x_test = x_test.drop(['target_class'], axis=1)
    y_test = y_test

    # 特征缩放：将特征数据分布调整为标准正态分布，均值为0，方差为1
    cols = x_train.columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=[cols])
    x_test = pd.DataFrame(x_test, columns=[cols])
    return x_train, y_train, x_test, y_test


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
    # pulsar = df[df['target_class'] == 1]
    # pulsar_count = pulsar["target_class"].value_counts()[1]
    # not_pulsar = df[df['target_class'] == 0]
    # not_pulsar_count = not_pulsar["target_class"].value_counts()[0]
    #
    # # pie plotting the stats between pulsars and not pulsars
    # plt.figure(figsize=(5, 5))
    # plt.pie(df["target_class"].value_counts().values, labels=["no-pulsar", "pulsar-stars"], autopct="%1.0f%%")
    # plt.title("Proportion of target variable in dataset")
    # plt.show()
    # print("There are " + str(pulsar_count) + " signals that belong to pulsar stars "
    #       + "and " + str(not_pulsar_count) + " signals that aren't from pulsars.")

    # features = df.iloc[:, 0:8]
    # plt.figure(figsize=(15, 20))
    # j = 0
    # for i in features:
    #     plt.subplot(4, 3, j + 1)
    #     sns.violinplot(x=df["target_class"], y=df[i], palette=["red", "lime"])
    #     plt.title(i)
    #     plt.axhline(df[i].mean(), linestyle="dashed", label="Mean value = " + str(round(df[i].mean(), 2)))
    #     plt.legend(loc="best")
    #     j = j + 1

    # show_view = cp.getboolean('Dataset', 'SHOW_VIEW')
    # if show_view:
    # show_data_set_summary(df)
    show_data_set_distribution(df)

    # plt.show()    # 直接阻塞显示窗口
    plt.pause(10)  # 等待10s后关闭所有显示窗口
    plt.close('all')
