
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
import util


def main():
    config = util.get_config("config.json")
    #preprocessing
    x,y = generate_dataset(config.features)
    x,y = under_sampling(x,y)
    x = x.fillna(x.mean())
    y = y.fillna(y.mean())
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.1)

    #grid search
    xgbc = xgb.XGBClassifier()
    paramaters = [
        {'max_depth' : [3,5,7]},
        {'min_child_weight':[0.8,0.9,1.0]},
        {'subsample':[0.8,0.9,1.0]},
        {'colsample_bytree':[0.8,0.9,1.0]},
    ]
    cv = GridSearchCV(xgbc,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)
    #best_forest = cv.best_estimator_
    cv.fit(train_x,train_y)
    pred = cv.predict(test_x)

    best = cv.best_estimator_
    print(best)
    print(accuracy_score(test_y,pred))
    print(classification_report(test_y,pred))
    importances = best.feature_importances_
    for f,i in zip(config.features,importances):
        print("{0:<20} : {1:.5f}".format(f,i))

def generate_dataset(features):
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect("output.db")

    f_orm = feature.Feature(db_con)
    target_columns = features
    for x,y in f_orm.fetch_horse(target_columns,race_type = "win"):
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = pd.DataFrame(dataset_x)
    dataset_y = pd.DataFrame(dataset_y)
    return dataset_x,dataset_y

def generate_dataset_races():
    dataset_x = []
    dataset_y = []

    config = util.get_config("config.json")
    db_con = sqlite3.connect("output.db")

    f_orm = feature.Feature(db_con)
    target_columns = config.features
    for x,y in f_orm.fetch_horse(target_columns):
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = pd.DataFrame(dataset_x)
    dataset_y = pd.DataFrame(dataset_y)
    return dataset_x,dataset_y


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
