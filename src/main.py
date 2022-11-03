import os
import sys
PATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH)

from src.deeplearning import pulsar, res14, cifar10, resimg14, cnn
from src.svm.gridsearch import grid_search
from src.svm.linear import svm_linear
from src.svm.polynomial import svm_polynomial
from src.svm.rbf import svm_rbf
from src.svm.sigmoid import svm_sigmoid
from src.utils.dataset import show_data_set_view, data_preprocess
from src.utils.performance import check_test_distribution
from src.utils.roc import roc, roc_auc


def svm():

    svm.x_train, svm.y_train, svm.x_test, svm.y_test =\
        data_preprocess()

    svm_linear()
    svm_rbf()
    svm_polynomial()
    svm_sigmoid()
    roc()
    roc_auc()
    grid_search()

    check_test_distribution()
    print("svm")

def deeplearning():
    # CNN : 一般全连接神经网络
    # pulsar.load_data()
    # cnn.info(2)
    # cnn.train(pulsar.trainloader, pulsar.testloader, PATH+"/model/pulsar.pth", 2)
    # model = pulsar.load_model(PATH+"/model/pulsar.pth")
    # dic = model.state_dict()
    # knn.pred(model, pulsar.testloader)

    # 残差网络处理cvs
    # pulsar.load_data()
    # res14.info(2)
    # res14.train(pulsar.trainloader, pulsar.testloader, PATH+"/model/pulsar-res.pth", 2)
    # model = pulsar.load_model(PATH+"/model/pulsar-res.pth")
    # dic = model.state_dict()
    # res14.pred(model, pulsar.testloader)

    # 残差网络处理图片
    cifar10.load_data()
    # resimg14.info(2)
    resimg14.train(cifar10.trainloader, cifar10.testloader, PATH+"/model/pulsar-img-res.pth", 2)
    # model = cifar10.load_model(PATH+"/model/pulsar-img-res.pth")
    # resimg14.pred(model, cifar10.testloader)


if __name__ == '__main__':
    # svm()
    deeplearning()
    print("main")
