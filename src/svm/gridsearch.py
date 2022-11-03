from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import src.svm as svm
from joblib import parallel_backend


def grid_search():
    svc = SVC()
    # declare parameters for hyperparameter tuning
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma':
                      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                  {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4],
                   'gamma': [0.01, 0.02, 0.03, 0.04, 0.05]}
                  ]
    with parallel_backend('threading', n_jobs=8):
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=0)
        grid_search.fit(svm.x_train, svm.y_train)
    print('GridSearch CV best score : {:.4f}'.format(grid_search.best_score_))
    # print parameters that give the best results
    print('Parameters that give the best results :', (grid_search.best_params_))
    # print estimator that was chosen by the GridSearch
    print('Estimator that was chosen by the search :', (grid_search.best_estimator_))
    print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(svm.x_test, svm.y_test)))