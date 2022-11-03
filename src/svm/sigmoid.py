from src.svm.train import svm_train


def svm_sigmoid():
    print("#############  sigmod kernel #############")
    kernel = 'sigmoid'
    svm_train(kernel, C=1.0)

    # svm_train(kernel, C=100.0)

    svm_train(kernel, C=10000.0)
