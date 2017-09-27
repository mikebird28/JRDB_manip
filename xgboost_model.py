
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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

    #preprocess
    train_x,train_y = dataset.races_to_horses(train_x,train_y)
    train_x = pd.DataFrame(train_x,columns = config.features)
    train_y = pd.DataFrame(train_y)
    train_x = dataset.fillna_mean(train_x)
    train_x = dataset.fillna_zero(train_x)
    train_x,train_y = under_sampling(train_x,train_y)

    test_rx = test_x
    test_ry = test_y

    test_hx,test_hy = dataset.races_to_horses(test_x,test_y)
    test_hx = pd.DataFrame(test_hx,columns = config.features)
    test_hy = pd.DataFrame(test_hy)
    test_hx = dataset.fillna_mean(test_hx)
    test_hx,test_hy = under_sampling(test_hx,test_hy)

    #grid search
    cv = xgb.XGBClassifier()
    """
    paramaters = [
        {'max_depth' : [7]},
        {'min_child_weight':[0.9]},
        {'subsample':[1.0]},
        {'colsample_bytree':[0.9,1.0]},
    ]
    cv = GridSearchCV(xgbc,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    """
    cv.fit(train_x,train_y)
    #best_forest = cv.best_estimator_
    cv.fit(train_x,train_y)
    evaluate.top_n_k(cv,config.features,test_rx,test_ry)
    pred = cv.predict(test_hx)

    #best = cv.best_estimator_
    #print(best)
    print(accuracy_score(test_hy,pred))
    print(classification_report(test_hy,pred))
    #importances = best.feature_importances_
    importances = cv.feature_importances_
    #evaluate.plot_importance(config.features,importances)
    evaluate.show_importance(config.features,importances)


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
