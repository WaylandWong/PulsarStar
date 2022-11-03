from src.svm.train import svm_train


# 线性
def svm_linear():
    print("############# linear kernel #############")

    kernel = 'linear'
    # svm_train(kernel=kernel, C=10)

    svm_train(kernel=kernel)

    # svm_train(kernel=kernel, C=1.0)

    # svm_train(kernel=kernel, C=100.0)

    # svm_train(kernel=kernel, C=1000.0)
