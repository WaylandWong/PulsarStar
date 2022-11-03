import src.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.utils.performance import performance, kfold


# C：惩罚系数，数值越高越容易过拟合
# gamma：是选择径向基函数（RBF）作为kernel后，该函数自带的一个参数
# gamma越大，支持向量越少，gamma越小，支持向量越多。数值越大越容易过拟合
def svm_train(kernel="rbf", C=1.0, gamma='auto'):
    svc = SVC(kernel=kernel, C=C, gamma=gamma)

    svc.fit(svm.x_train, svm.y_train)

    svm.y_pred_train = svc.predict(svm.x_train)
    train_score = accuracy_score(svm.y_train, svm.y_pred_train)

    # 预测
    svm.y_pred = svc.predict(svm.x_test)
    test_score = accuracy_score(svm.y_test, svm.y_pred)
    print('kernel:{0}, C:{1}, gamma:{2} hyper parameters'
          .format(kernel, C, gamma))
    print('Train Model accuracy score: {:0.4f}'
          .format(train_score))
    print('Train Model accuracy score: {:0.4f}'
          .format(test_score))

    kfold(svc, svm.x_train, svm.y_train, svm.x_test, svm.y_test)
    performance(['0', '1'], svm.y_test, svm.y_pred)

