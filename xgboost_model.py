
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb
import numpy as np
import pandas as pd
import sqlite3
import feature
import dataset
import util
import evaluate


def main():
    config = util.get_config("config/config.json")
    #generate dataset
    db_path = "db/output_v3.db"
    x,y = dataset.load_races(db_path,config.features,"win")
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.1)
    pca = PCA(n_components = 20)

    #preprocess for training data
    ptrain_x,ptrain_y = dataset_for_pca(trian_x,train_y)
    ptrain_x = dataset.fillna_mean(ptrain_x)

    pca.fit(ptrain_x)
    pca_train_x = pca.transform(ptrain_x)
    print(pca.explained_variance_ratio_)

    train_x = dataset.add_race_info(train_x,pca_train_x)
    train_x,train_y = dataset.races_to_horses(train_x,train_y)
    train_x,train_y = preprocess(train_x,train_y)

    #preprocess for test data
    test_rx = test_x
    test_ry = test_y
    test_rx = dataaset.fillna(test_rx)
    ptest_rx,ptest_ry = dataset_for_pca(test_rx,test_ry)
    ptest_rx = pca.transform(ptest_rx)

    test_hx,test_hy = dataset.races_to_horses(test_x,test_y,ptest_rx)
    test_hx,test_hy = preprocess(test_hx,test_hy,config.features)

    #grid search
    xgbc = xgb.XGBClassifier()

    paramaters = [
        {'max_depth' : [10]},
        {'min_child_weight':[0.6,0.8,1.0]},
        {'subsample':[0.6,0.8,1.0]},
        {'colsample_bytree':[0.6,0.8,1.0]},
    ]
    cv = GridSearchCV(xgbc,paramaters,cv = 3,scoring='accuracy',verbose = 2)

    #best_forest = cv.best_estimator_
    #cv.fit(train_x,train_y)
    xgbc.fit(train_x,train_y)
    #pred = cv.predict(test_hx)
    pred = xgbc.predict(test_hx)

    #best = cv.best_estimator_
    #print(best)

    #print("Top 1   : {0}".format(evaluate.top_n_k(xgbc,config.features,test_rx,test_ry)))
    #print("Top 1   : {0}".format(evaluate.top_n_k(cv,config.features,test_rx,test_ry)))
    print("Accuracy: {0}".format(accuracy_score(test_hy,pred)))
    print(classification_report(test_hy,pred))
    #importances = best.feature_importances_
    importances = xgbc.feature_importances_
    #evaluate.plot_importance(config.features,importances)
    evaluate.show_importance(config.features,importances)

def dataset_for_pca(x,y):
    p_x,p_y = dataset.pad_race(x,y)
    p_x = [join_list(child) for child in ptrain_x]
    return p_x,p_y

def preprocess(x,y,features):
    x,y = under_sampling(x,y)
    return (x,y)

def join_list(ls):
    result = []
    for child in ls:
        result += child
    return result

if __name__=="__main__":
    main()
