from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

def kfold_cross_validation(feature, label, k, classifier, reg_para,
                           report_confidence=False):
    """ kfold cross-validation for linear neural decoding.

        Inputs:
            - features: features for decoding.
            - label: label for decoding.
            - k: number of folds.
            - classifer: selected classifier.
            - reg_para: regularization parameter.
            - report_confidence: if True, return the predicted confidence on
                            the correct class instead of accuracy./
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

        if not report_confidence:
            avg_acc.append(cls.score(test_feat, test_label))
        else:
            raw_pred = cls.decision_function(test_feat)
            # scale the decision function into confidence
            training_df = cls.decision_function(train_feat)
            scale_min, scale_max = training_df.min(), training_df.max()
            pred_conf = (raw_pred-scale_min)/(scale_max-scale_min)
            
            adjusted_conf = [pred_conf[idx] if test_label[idx]==1 else 1-pred_conf[idx]
                        for idx in range(len(pred_conf))]
            avg_acc.append(np.mean(adjusted_conf))
    
    return np.mean(avg_acc)