from src.svm.train import svm_train


def svm_polynomial():
    print("#############  polynomial kernel #############")
    kernel = 'poly'
    svm_train(kernel, C=1.0)

    svm_train(kernel, C=100.0)

    svm_train(kernel, C=1000.0)
