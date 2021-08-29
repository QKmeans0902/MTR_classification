import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve

fpath = 'path/for/feature/file/(csv)'
lpath = 'path/for/label/file/(csv)'
seed = 81 # random seed
no_folds = 10 # number of folds in out_loop
no_nested_folds = 10 # number of folds in nested_loop

skf = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=101)
nested_skf = StratifiedKFold(n_splits=no_nested_folds, shuffle=True)
param_grid = {'C': np.logspace(-4, 3, 8)}
eval_metrics = np.zeros((skf.n_splits, 4))
print('Loading data ...')
features = np.loadtxt(fpath, delimiter=',')
labels = np.loadtxt(lpath, dtype='int32')
print('Finished')

# ROC plotting preparation
TPR, AUC = [], []
mean_fpr = np.linspace(0, 1, 100)

for n_cv, (train_ind, test_ind) in enumerate(skf.split(features, labels)):
    print('Processing the No.%i cross-validation in %i-fold CV' % (n_cv + 1, skf.n_splits))
    x_train, y_train = features[train_ind, ], labels[train_ind, ]
    x_test, y_test = features[test_ind, ], labels[test_ind, ]
    
    # Training
    init_clf = SVC(kernel='linear')
    grid = GridSearchCV(init_clf, param_grid, cv=nested_skf, scoring='accuracy', n_jobs=5)
    grid.fit(x_train, y_train)
    print('----The best parameter C: %.2e with accuracy of %f' % (grid.best_params_['C'], grid.best_score_))
    clf = SVC(kernel='linear', C=grid.best_params_['C'], probability=True)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)
    
    # ROC information for each fold
    cv_fpr, cv_tpr, cv_thresholds = roc_curve(y_test, y_proba[:, 1])
    cv_auc = auc(cv_fpr, cv_tpr)
    interp_tpr = np.interp(mean_fpr, cv_fpr, cv_tpr)
    interp_tpr[0] = 0.0
    TPR.append(interp_tpr)
    AUC.append(cv_auc)
    
    # Evaluation
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    cv_accuracy = (tn + tp) / (tn + fp + fn + tp) 
    cv_sensitivity = tp / (tp + fn)
    cv_specificity = tn / (tn + fp)
    eval_metrics[n_cv, 0] = cv_accuracy
    eval_metrics[n_cv, 1] = cv_sensitivity
    eval_metrics[n_cv, 2] = cv_specificity
    eval_metrics[n_cv, 3] = cv_auc
    
# reporting model evaluation measures
df = pd.DataFrame(eval_metrics)
df.columns = ['ACC', 'SEN', 'SPE', 'AUC']
df.index = ['CV_' + str(i + 1) for i in range(skf.n_splits)]
print(df)
print('\nAverage Accuracy: %.4f' % (eval_metrics[:, 0].mean()))
print('Average Sensitivity: %.4f' % (eval_metrics[:, 1].mean()))
print('Average Specificity: %.4f' % (eval_metrics[:, 2].mean()))
print('Average area under ROC curve: %.4f' % (eval_metrics[:, 3].mean()))

# saving ROC plotting information
mean_tpr = np.mean(TPR, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
np.savez(fpath + '/../ROC_MTR.npz', tpr=mean_tpr, fpr=mean_fpr, auc=mean_auc)
