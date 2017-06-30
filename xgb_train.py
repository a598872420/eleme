# coding=utf-8
import sys
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import pandas as pd

feature_name_file = sys.argv[3]
tag = sys.argv[4]
gen_dir = "generated"

names = []
for line in open(sys.argv[5]):
    names.append(line.strip().split("\t")[0])
names.append("similarity")
names.append("label")
df = pd.read_table(sys.argv[6], names=names)
#print df.iloc[0]

#sys.exit(0)

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

feature_names = []
for line in open(sys.argv[3]):
    feature_names.append(line.strip().split("\t")[1])
print 'feature length:', len(feature_names)

X, y = load_svmlight_file(sys.argv[1])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
print X_train.shape
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
#dtrain = xgb.DMatrix(sys.argv[1], feature_names=feature_names)
dtest = xgb.DMatrix(sys.argv[2], feature_names=feature_names)
#dvalid = xgb.DMatrix(sys.argv[2])#, feature_names=feature_names)
print dtrain.num_col(), dtest.num_col()
#TODO 2
num_round = 1500
#num_round = 10

# specify parameters via map
train_label = dtrain.get_label()
ratio = float(np.sum(train_label == 0)) / np.sum(train_label == 1)
print 'scale_pos_weight:', ratio

param = {'max_depth': 10,
         'eta': 0.02,
         'silent': 1,
         'objective': 'binary:logistic',
         'booster': 'gbtree',
         'gamma': 0.1,
         'nthread': 20,
         'scale_pos_weight': ratio,
         'eval_metric': ['auc', 'error'],#, 'error@0.8', 'error@0.9'],
         'n_estimators': 1500
         #'early_stopping_rounds': 100,
         #'lambda':550,
         #'subsample':0.7,
         #'colsample_bytree':0.4,
         #'min_child_weight':3,
         }
evallist = [(dtrain, 'train'), (dvalid, 'valid')]

bst = xgb.train(param, dtrain, num_round, evallist)

f, ax = plt.subplots(1, 1, figsize=(20, 15))
plt.subplots_adjust(left=0.3, right=0.9)
importance_fig = xgb.plot_importance(bst, ax=ax)
plt.savefig('{}/importance_{}.png'.format(gen_dir, tag))
bst.save_model('{}/xgboost_{}.model'.format(gen_dir, tag))

test_score = bst.predict(dtest)
df.loc[:,'score'] = test_score

test_label = dtest.get_label()
auc_score = sklearn.metrics.roc_auc_score(test_label, test_score)
print 'test auc:', auc_score

all = dtest.num_row()
positive = np.sum(test_label == 1)
negative = all - positive
print 'test size: %s (positive:%s, negative:%s)' % (all, positive, negative)

def test(threshold):
    predict_label = (test_score > threshold).astype(int)
    matrix = sklearn.metrics.confusion_matrix(test_label, predict_label)
    true_positive = matrix[1][1]
    true_negative = matrix[0][0]
    print '*' * 80
    print 'threshold:', threshold
    print 'positive accuracy: %s (%s / %s)' % (true_positive * 1.0 / positive, true_positive, positive)
    print 'negative accuracy: %s (%s / %s)' % (true_negative * 1.0 / negative, true_negative, negative)
    print matrix
    precision = sklearn.metrics.precision_score(test_label, predict_label)
    print 'precision_score:', precision
    recall = sklearn.metrics.recall_score(test_label, predict_label)
    print 'recall_score:', recall
    f1 = sklearn.metrics.f1_score(test_label, predict_label)
    print 'f1:', f1
    print '-' * 80

#TODO 3
test(0.5)
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    test(i)

predict_label = (test_score > 0.5).astype(int)
true_positive_index = np.nonzero((predict_label == 1) & (test_label == 1))[0]
false_negative_index = np.nonzero((predict_label == 0) & (test_label == 1))[0]
false_positive_index = np.nonzero((predict_label == 1) & (test_label == 0))[0]
true_negative_index = np.nonzero((predict_label == 0) & (test_label == 0))[0]

size = 10
true_positive_index_subset = true_positive_index[0:size]
false_negative_index_subset = false_negative_index[0:size]
false_positive_index_subset = false_positive_index[0:size]
true_negative_index_subset = true_negative_index[0:size]

def print_df(df):
    for row in df.itertuples():
        a, b, c = row[1:4]
        print '%s,%s\t%s' % (a.astype('int64'), b.astype('int64'), c)

def show_cases():
    print true_positive_index_subset
    print false_negative_index_subset
    print 'true positive'
    print_df(df.iloc[true_positive_index_subset].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']])
    #print df.iloc[true_positive_index_subset].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']]
    print 'false negative'
    print_df(df.iloc[false_negative_index_subset].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']])
    print 'false positive'
    print_df(df.iloc[false_positive_index_subset].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']])
    print 'true negative'
    print_df(df.iloc[true_negative_index_subset].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']])

print 'same tracking id score:'
print df[df['tracking_id'] == df['tracking_id_loaded']][0:size].loc[:, ['tracking_id', 'tracking_id_loaded', 'score']]


average_precision = sklearn.metrics.average_precision_score(test_label, test_score)
print 'average_precision_score:', average_precision

precision, recall, thresholds = sklearn.metrics.precision_recall_curve(test_label, test_score)

f1 = 2 * np.divide(np.multiply(precision, recall), precision + recall)
thresholds = np.append(thresholds, 1.0)

def plot():
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(thresholds, precision, color='r', label='Precision curve')
    plt.plot(thresholds, recall, color='g', label='Recall curve')
    plt.plot(thresholds, f1, color='b', label='f1 curve')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(auc_score))
    plt.legend(loc="lower left")
    plt.savefig("{}/prf1_{}.png".format(gen_dir, tag))

show_cases()
plot()
