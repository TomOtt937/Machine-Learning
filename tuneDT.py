''' Sub-routines for tuning Decision Tree algorithm '''
import sys
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def test_precision(clf, dataset, feature_list, folds = 1000):
    # return precision using StratifiedShuffleSplit to evaluate
    # code adapted from tester.py
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        return 1.0*true_positives/(true_positives+false_positives)  #precision

    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        return 0

def maximize_parameter(my_dataset, features_list):
    # Return min_samples_split that maximizes precision for Decision Tree
    best_precision = 0
    best_i = 0
    for i in range(2, 21):
        clf = tree.DecisionTreeClassifier(min_samples_split=i)
        precision = test_precision(clf, my_dataset, features_list)
        if precision > best_precision:
            best_precision = precision
            best_i = i
    return best_i, best_precision
