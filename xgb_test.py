# coding=utf-8
import sys
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

test_feature_file = sys.argv[1]
test_sample_file = sys.argv[2]
feature_name_file = sys.argv[3]
feature_map_file = sys.argv[4]
tag = sys.argv[5]
tracking_id = sys.argv[6]
tracking_id_loaded = sys.argv[7]
gen_dir = "generated"

names = []
for line in open(feature_name_file):
    names.append(line.strip().split("\t")[0])
names.append("similarity")
names.append("label")
df = pd.read_table(test_sample_file, names=names)

feature_names = []
for line in open(feature_map_file):
    feature_names.append(line.strip().split("\t")[1])
print 'feature length:', len(feature_names)

dtest = xgb.DMatrix(test_feature_file, feature_names=feature_names)

bst = xgb.Booster(model_file='{}/xgboost_{}.model'.format(gen_dir, tag))
test_score = bst.predict(dtest)
df.loc[:,'score'] = test_score

import make_feature
fmap, nmap = make_feature.loadfmap( 'similarity.fmap' )
print tracking_id, tracking_id_loaded
row = df[((df['tracking_id'] == int(tracking_id)) & (df['tracking_id_loaded'] == int(tracking_id_loaded)))]
new_map = sorted(nmap.items(), key = lambda d: d[1]['name'])
for i in range( len(new_map) ):
    item = new_map[i][1]
    if item['is_used'] == '1':
        name = item['name']
        if item['type'] == 'i':
            sample_name, v = name.split("=")
            value = row.loc[:, sample_name].values[0]
            if int(v) == int(value):
                print '%s\t%s' % (item['name'], '1')
            else:
                print '%s\t%s' % (item['name'], '0')
        else:
            sample_name = name
            value = row.loc[:, sample_name].values[0]
            print '%s\t%s' % (item['name'], value)
print 'score:', row['score'].values[0]
