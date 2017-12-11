#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
# poi_mutual_correspondence
# Reflects if a person is sending a similar number of emails to POIs as they
# receive from POIs, and multiplies that proportion by the number of email
# sent to POIs.  The more emails a person sends to POIs when they are receiving
# a similar number of responses could indicate a common interest.
for person in data_dict:
    if data_dict[person]['from_poi_to_this_person'] != 'NaN':
        if data_dict[person]['from_poi_to_this_person'] == 0:
            data_dict[person]['poi_mutual_correspondence'] = 0
        else:
            if data_dict[person]['from_this_person_to_poi'] < \
               data_dict[person]['from_poi_to_this_person']:
                data_dict[person]['poi_mutual_correspondence'] =  \
                  round(data_dict[person]['from_this_person_to_poi']* \
                    (float(data_dict[person]['from_this_person_to_poi'])/ \
                     data_dict[person]['from_poi_to_this_person']), 2)
            else:
                data_dict[person]['poi_mutual_correspondence'] = \
                  round(data_dict[person]['from_this_person_to_poi']* \
                    (float(data_dict[person]['from_poi_to_this_person'])/ \
                     data_dict[person]['from_this_person_to_poi']), 2)
    else:
        data_dict[person]['poi_mutual_correspondence'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
clf = tree.DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_list = ['poi','bonus','poi_mutual_correspondence']
from tuneDT import maximize_parameter
best_parm, best_precision = maximize_parameter(my_dataset, features_list)
print "Tuned Decision Tree for features %s uses min_samples_split=%i" % \
       (features_list, best_parm)
clf = tree.DecisionTreeClassifier(min_samples_split=best_parm)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
clf.fit(features, labels)
importances = clf.feature_importances_
print "Feature Importances: %s (%0.4f), %s (%0.4f)" % (features_list[1], \
                       importances[0], features_list[2], importances[1])
test_classifier(clf, my_dataset, features_list)

best_parm_previous = best_parm
best_precision_previous = best_precision
features_list_previous = features_list

features_list = ['poi','bonus']
best_parm, best_precision = maximize_parameter(my_dataset, features_list)
print "Tuned Decision Tree for features %s uses min_samples_split=%i" % \
       (features_list, best_parm)
clf = tree.DecisionTreeClassifier(min_samples_split=best_parm)
test_classifier(clf, my_dataset, features_list)

print
if best_precision > best_precision_previous:
    print "Keeping feature list %s with min_samples_split=%i" % \
           (features_list, best_parm)
    clf = tree.DecisionTreeClassifier(min_samples_split=best_parm)
else:
    print "Keeping feature list %s with min_samples_split=%i" % \
           (features_list_previous, best_parm_previous)
    clf = tree.DecisionTreeClassifier(min_samples_split=best_parm_previous)
    features_list = features_list_previous

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
