
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
import xgboost as xgb
#import numpy as np
import pandas as pd
import sqlite3
import argparse
import dataset2
import util
import evaluate


CACHE_PATH = "./cache/xgbc"
MODEL_PATH = "./models/xgbc"
pd.options.display.max_rows = 1000
past_n = 3
predict_type = "is_win"

def main(use_cache = False):
    predict_type = "is_win"
    config = util.get_config("config/xgbc_config2.json")
    db_path = "db/output_v15.db"
    db_con = sqlite3.connect(db_path)

    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    xgbc(config.features,datasets)
    #xgbc_wigh_bayessearch(config.features,datasets)


def generate_dataset(predict_type,db_con,config):
    print(">> loading dataset")
    main_features = config.features

    where = "info_year > 08 and info_year < 90"
    x,y = dataset2.load_dataset(db_con,main_features,["is_win","win_payoff","is_place","place_payoff"],where = where)
    """
    categorical_dic = dataset2.nominal_columns(db_con)
    dummy_col = dataset2.dummy_column(x,categorical_dic)
    x = dataset2.get_dummies(x,categorical_dic)
    """
    main_features = sorted(x.columns.values.tolist())
    main_features_dropped = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)
    """

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = dummy_col)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = dummy_col)
    """

    test_x = test_x.loc[:,main_features]
    test_rx,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_x,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )

    print(">> under sampling train dataset")
    train_x,train_y = dataset2.under_sampling(train_x,train_y,key = predict_type,magnif = 3)
    train_x = train_x.drop("info_race_id",axis = 1)
    train_x = train_x.loc[:,main_features_dropped]

    print(">> under sampling train dataset")
    test_x,test_y = dataset2.under_sampling(test_x,test_y,key = predict_type)
    test_x = test_x.drop("info_race_id",axis = 1)
    test_x = test_x.loc[:,main_features_dropped]

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
        "test_rx"      : test_rx,
        "test_r_win"   : test_r_win,
        "test_rp_win"   : test_rp_win,
        "test_r_place" : test_r_place,
        "test_rp_place" : test_rp_place,
    }
    return datasets


def xgbc(features,datasets):
    train_x = datasets["train_x"]
    features = train_x.columns
    train_y = datasets["train_y"].loc[:,predict_type]
    test_x  = datasets["test_x"]
    test_y  = datasets["test_y"].loc[:,predict_type]
    test_rx = datasets["test_rx"]
    #test_ry = datasets["test_ry"]
    test_r_win = datasets["test_r_win"]
    test_r_place = datasets["test_r_place"]
    test_rp_win = datasets["test_rp_win"]
    test_rp_place = datasets["test_rp_place"]
    
    xgbc = xgb.XGBClassifier(
        n_estimators = 300,
        colsample_bytree =  0.869719614599,
        learning_rate =  0.001,
        min_child_weight = 0.42206733097,
        subsample = 0.99637221573,
        max_depth = 10,
        gamma =  0.90110124545,
    )
    """
    xgbc = xgb.XGBClassifier(
        n_estimators = 100,
        colsample_bytree =  0.8,
        learning_rate =  0.001,
        min_child_weight = 1.55,
        subsample = 0.66,
        max_depth = 10,
        gamma =  0.71,
    )
    """
 
    xgbc.fit(train_x,train_y)

    pred = xgbc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    """
    win_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
    """

    report = classification_report(test_y,pred)
    print(report)

    importances = xgbc.feature_importances_
    evaluate.show_importance(features,importances)

def xgbc_wigh_gridsearch(features,datasets):
    train_x = datasets["train_x"]
    train_y = datasets["train_y"]
    test_x  = datasets["test_x"]
    test_y  = datasets["test_y"]
    test_rx = datasets["test_rx"]
    test_ry = datasets["test_ry"]
    test_r_win = datasets["test_r_win"]
    test_r_place = datasets["test_r_place"]
    test_rp_win = datasets["test_rp_win"]
    test_rp_place = datasets["test_rp_place"]


    paramaters = [
        {'n_estimators':[500],
        'learning_rate':[0.05,0.1],
        'max_depth' : [5,7],
        'subsample':[0.6,0.7],
        'min_child_weight':[1.0,1,1,1.2,1,3],
        'colsample_bytree':[0.8,1.0]},
    ]

    xgbc = xgb.XGBClassifier()
    cv = GridSearchCV(xgbc,paramaters,cv = 3,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    win_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_win,test_rp_win)
    place_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_place,test_rp_place)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    print("Paramaters")
    #best_parameters, score, _ = max(cv.grid_scores_, key=lambda x: x[1])    
    best_parameters = cv.best_params_
    print(best_parameters)
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))

    print("")
    print("Features")
    best = cv.best_estimator_
    importances = best.feature_importances_
    evaluate.show_importance(features,importances)

def xgbc_wigh_bayessearch(features,datasets):
    train_x = datasets["train_x"]
    features = train_x.columns
    train_y = datasets["train_y"].loc[:,predict_type]
    test_x  = datasets["test_x"]
    test_y  = datasets["test_y"].loc[:,predict_type]
    test_rx = datasets["test_rx"]
    #test_ry = datasets["test_ry"]
    test_r_win = datasets["test_r_win"]
    test_r_place = datasets["test_r_place"]
    test_rp_win = datasets["test_rp_win"]
    test_rp_place = datasets["test_rp_place"]
 
    paramaters = [
        {
        'max_depth' : (3,10),
        'subsample':(0.1,1.0),
        'min_child_weight':(0.3,2.0),
        'colsample_bytree':(0.3,1.0),
        'gamma':(0,1.0)}
    ]

    xgbc = xgb.XGBClassifier(n_estimators = 100,learning_rate = 0.001)
    cv = BayesSearchCV(xgbc,paramaters,cv = 3,scoring='accuracy',n_iter = 30,verbose = 3)
    cv.fit(train_x,train_y)
    pred = cv.predict(test_x)

    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    """
    win_eval  = evaluate.top_n_k(cv,test_rx,test_r_win,test_rp_win)
    place_eval  = evaluate.top_n_k(cv,test_rx,test_r_place,test_rp_place)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
    """

    print("Paramaters")
    #best_parameters, score, _ = max(cv.grid_scores_, key=lambda x: x[1])    
    best_parameters = cv.best_params_
    print(best_parameters)
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))

    print("")
    print("Features")
    best = cv.best_estimator_
    importances = best.feature_importances_
    evaluate.show_importance(features,importances)

def dataset_for_pca(x,y,mean = None,std = None):
    x = dataset2.normalize(x,mean = mean,std = std)
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race(x)
    return (x,y)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
