from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearn.svm import LinearSVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def kfold_cross_validation(feature, label, k, classifier, reg_para,
                           report_confidence=False, category_mapping=None):
    """ kfold cross-validation for linear neural decoding.

        Inputs:
            - features: features for decoding.
            - label: label for decoding.
            - k: number of folds.
            - classifer: selected classifier.
            - reg_para: regularization parameter.
            - report_confidence: if True, return the predicted confidence on
                            the correct class instead of accuracy.
            - category_mapping if not None, return a dictionary storing the accuracy for different
                category.
        Return:
            Decoding performance in accuracy.
    """

    kf = KFold(n_splits=k)
    if category_mapping is None:
        avg_acc = [] 
    else:
        avg_acc = dict()
        for k in np.unique(list(category_mapping.values())):
            avg_acc[k] = []
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

        if category_mapping is None:
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
        else:
            pred_cls = cls.predict(test_feat)
            correctness = pred_cls == test_label
            for k in avg_acc:
                current_pool = [idx for idx in range(len(correctness)) if category_mapping[test_label[idx]]==k]
                avg_acc[k].append(np.mean(correctness[np.array(current_pool)]))

    if category_mapping is None:
        return np.mean(avg_acc)
    else:
        for k in avg_acc:
            avg_acc[k] = np.mean(avg_acc[k])
        return avg_acc
    

def kfold_cross_validation_regression(feature, label, k, reg_para, regressor, kernel='rbf'):
    """ kfold cross-validation for linear neural decoding for regression.

        Inputs:
            - features: features for decoding.
            - label: label for decoding.
            - k: number of folds.
            - reg_para: regularization parameter.
            - regressor: model used for regression

        Return:
            A dictionary storing the ground truth and prediction for each test sample
    """

    kf = KFold(n_splits=k)
    record = {'gt': [], 'pred': []}

    for i, (train_index, test_index) in enumerate(kf.split(feature)):
        train_feat = np.array([feature[idx] for idx in train_index])
        train_label = np.array([label[idx] for idx in train_index])
        test_feat = np.array([feature[idx] for idx in test_index])
        test_label = np.array([label[idx] for idx in test_index])

        if regressor == 'ridge':
            cls = Ridge(alpha=reg_para).fit(train_feat, train_label)
        elif regressor == 'svr':
            cls = MultiOutputRegressor(SVR(kernel=kernel, C=reg_para, max_iter=1500)).fit(train_feat, train_label)
        elif regressor == 'random_forest':
            cls = RandomForestRegressor(n_estimators=100).fit(train_feat, train_label)
        else:
            NotImplementedError('Selected model not supported')

        pred_cls = cls.predict(test_feat)
        record['gt'].extend(test_label)
        record['pred'].extend(pred_cls)

    record['gt'] = np.array(record['gt'])
    record['pred'] = np.array(record['pred'])

    return record