
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

    #x,y = preprocess_for_races(config.features,x,y)
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.1)

    ptrain_x,ptrain_y = dataset.pad_race(train_x,train_y)
    ptrain_x = [join_list(child) for child in ptrain_x]
    ptrain_x = pd.DataFrame(ptrain_x)

    pca = PCA(n_components = 20)
    pca.fit(ptrain_x)
    pca_train_x = pca.transform(ptrain_x)

    print(pca.explained_variance_ratio_)
    print("OK")

    #preprocess
    train_x,train_y = dataset.races_to_horses(train_x,train_y,pca_train_x)
    config.features = [str(i) for i in range(76)]

    #train_x,train_y = dataset.races_to_horses(train_x,train_y)
    train_x = pd.DataFrame(train_x,columns = config.features)
    train_y = pd.DataFrame(train_y)
    train_x = dataset.fillna_mean(train_x)
    train_x = dataset.fillna_zero(train_x)
    train_x,train_y = under_sampling(train_x,train_y)

    test_rx = test_x
    test_ry = test_y

    ptest_rx,ptest_ry = dataset.pad_race(test_rx,test_ry)
    ptest_rx = [join_list(child) for child in ptest_rx]
    ptest_rx = pd.DataFrame(ptest_rx)
    ptest_rx = pca.transform(ptest_rx)

    test_hx,test_hy = dataset.races_to_horses(test_x,test_y,ptest_rx)
    test_hx = pd.DataFrame(test_hx,columns = config.features)
    test_hy = pd.DataFrame(test_hy)
    test_hx = dataset.fillna_mean(test_hx)
    test_hx = dataset.fillna_zero(test_hx)

    test_hx,test_hy = under_sampling(test_hx,test_hy)

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

def preprocess_for_races(feature,rx,ry):
    hx,hy = dataset.races_to_horses(rx,ry)
    result_x = []
    result_y = []

    count = 0
    for x,y in zip(rx,ry):
        if count%100 == 0:
            print(count)
        pd_x = pd.DataFrame(x,columns = feature)
        pd_x = dataset.fillna_mean(pd_x)
        pd_x = dataset.fillna_zero(pd_x)

        mean = pd_x.mean()
        std = pd_x.std().clip(lower=1e-4)
        pd_x = (pd_x - pd_x.mean())/std
        x = pd_x.values.tolist()
        
        result_x.append(x)
        result_y.append(y)
        count += 1
    return (result_x,result_y)

def join_list(ls):
    result = []
    for child in ls:
        result += child
    return result
        
def under_sampling(x,y):
    con = pd.concat([y,x],axis = 1)

    loweset_frequent_value = 1
    low_frequent_records = con.ix[con.iloc[:,0] == 1,:]
    other_records = con.ix[con.iloc[:,0] != 1,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.iloc[:,1:]
    con_y = con.iloc[:,0]
    return con_x,con_y

if __name__=="__main__":
    main()
