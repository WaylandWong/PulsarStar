import itertools

import numpy as np
from joblib import parallel_backend
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score


# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.pause(10)  # 等待10s后关闭所有显示窗口
    plt.close('all')


# Stratified k-fold Cross Validation with shuffle split
def kfold(svc, x_train, y_train, x_test, y_test):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    with parallel_backend('threading', n_jobs=8):
        kfold_train_scores = cross_val_score(svc, x_train, y_train, cv=kfold)
        kfold_test_scores = cross_val_score(svc, x_test, y_test, cv=kfold)

    print('Train Model cross-validation scores:\n {}'
          .format(kfold_train_scores))
    print('Test Model cross-validation scores :\n {}'
          .format(kfold_test_scores))

    print('Train Model Average stratified  cross-validation scores {:0.4f} '
          .format(kfold_train_scores.mean()))
    print('Test Model Average stratified cross-validation scores: {:0.4f} '
          .format(kfold_test_scores.mean()))


# 模型指标评价
def performance(class_names, y_test, y_pred, show_confusion=False):
    # 混淆矩阵
    # class_names = np.array(['0', '1'])
    with parallel_backend('threading', n_jobs=8):
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        if show_confusion:
            plot_confusion_matrix(cm, class_names)

    if cm.size < 4:
        return
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    classification_accuracy = 0
    classification_error = 0
    precision = 0
    recall = 0
    false_positive_rate = 0
    specificity = 0
    # accuracy 精度 精度是分类正确的样本数占样本总数的比例
    if TP + TN + FP + FN != 0:
        classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
        # error 错误率 错误率是分类错误的样本数占样本总数的比例
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        if TP + FP != 0:
            # precision 查准率 正确预测的正样本数占所有预测为正样本的数量的比值
            precision = TP / float(TP + FP)
        if TP + FN != 0:
            # recall(true positive rate) 查全率/召回率 正确预测的正样本数占真实正样本总数的比值
            recall = TP / float(TP + FN)
        if TN + FP != 0:
            # 假阳性率
            false_positive_rate = FP / float(FP + TN)

            specificity = TN / (TN + FP)
    if show_confusion:
        # print('Confusion matrix', cm)
        print('True Positives(TP) = ', TP)
        print('True Negatives(TN) = ', TN)
        print('False Positives(FP) = ', FP)
        print('False Negatives(FN) = ', FN)

        print(classification_report(y_test, y_pred))
        print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
        print('Classification error : {0:0.4f}'.format(classification_error))
        print('Precision : {0:0.4f}'.format(precision))
        print('Recall or Sensitivity : {0:0.4f}'.format(recall))
        print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
        print('Specificity : {0:0.4f}'.format(specificity))

    return classification_accuracy, precision, recall


def check_test_distribution(y_test):
    counts = y_test.value_counts()
    null_accuracy = (counts[0] / (counts[0] + counts[1]))

    print('Null accuracy score: {0:0.4f}'.format(null_accuracy))
