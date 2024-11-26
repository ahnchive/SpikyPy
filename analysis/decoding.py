from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

def kfold_cross_validation(feature, label, k, classifier, reg_para):
    """ kfold cross-validation for linear neural decoding.

        Inputs:
            - features: features for decoding.
            - label: label for decoding.
            - k: number of folds.
            - classifer: selected classifier.
            - reg_para: regularization parameter.
        
        Return:
            Decoding performance in accuracy.
    """

    kf = KFold(n_splits=k)
    avg_acc = []
    for i, (train_index, test_index) in enumerate(kf.split(feature)):
        train_feat = np.array([feature[idx] for idx in train_index])
        train_label = np.array([label[idx] for idx in train_index])
        test_feat = np.array([feature[idx] for idx in test_index])
        test_label = np.array([label[idx] for idx in test_index])

        if classifier == 'logistic':
            cls = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
        elif classifier == 'svc':
            cls = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
        else:
            raise NotImplementedError(
                "Support for the selected classifier is not implemented yet")

        avg_acc.append(cls.score(test_feat, test_label))
    
    return np.mean(avg_acc)