# Receiver Operating Characteristic Curve
# at various classification threshold levels

# plot ROC Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def history(data):

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.plot(data['loss'], label='Loss')
    plt.title('Loss Function evolution during training')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(data['fn'], label='fn')
    plt.plot(data['val_fn'], label='val_fn')
    plt.title('Accuracy evolution during training')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(data['precision'], label='precision')
    plt.plot(data['val_precision'], label='val_precision')
    plt.title('Precision evolution during training')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(data['recall'], label='recall')
    plt.plot(data['val_recall'], label='val_recall')
    plt.title('Recall evolution during training')
    plt.legend()

    plt.show()

def roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.rcParams['font.size'] = 12

    plt.title('ROC curve for Predicting a Pulsar Star classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.show()


def roc_auc(x_train, y_train, y_test, y_pred):
    # compute ROC AUC
    ROC_AUC = roc_auc_score(y_test,y_pred)

    print('ROC AUC : {:.4f}'.format(ROC_AUC))

    linear_svc = SVC(kernel='linear', C=1.0)
    Cross_validated_ROC_AUC = cross_val_score(linear_svc, x_train, y_train, cv=10, scoring='roc_auc').mean()

    print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))