from src.svm.train import svm_train


# 径向核函数
def svm_rbf():
    print("############# rbf kernel #############")

    kernel = 'rbf'
    svm_train(kernel=kernel)

    svm_train(kernel=kernel, gamma=0.1)

    svm_train(kernel=kernel, gamma=0.5)

    svm_train(kernel=kernel, gamma=0.9)

    svm_train(kernel=kernel, gamma=1)

    svm_train(kernel=kernel, gamma=100)

    svm_train(kernel=kernel, gamma=1000)
