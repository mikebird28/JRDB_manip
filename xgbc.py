# -*- coding:utf-8 -*-
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
import xgboost as xgb
import pandas as pd
import numpy as np
import data_processor,evaluate,util
import argparse


CACHE_PATH = "./cache/xgbc"
MODEL_PATH = "./models/xgbc"
pd.options.display.max_rows = 1000
past_n = 3
predict_type = "is_win"

def main(use_cache = False):
    config = util.get_config("config/xgbc_config2.json")
    db_path = "db/output_v18.db"
    if use_cache:
        print("[*] load dataset from cache")
        dp = data_processor.load_from_cache(CACHE_PATH)
    else:
        print("[*] load dataset from database")
        dp = generate_dataset(db_path,config)
        dp.save(CACHE_PATH)
    xgbc(config.features,dp)
    #xgbc_wigh_bayessearch(config.features,datasets)

def generate_dataset(db_path,config):
    x_columns = config.features
    #y_columns = ["is_win","is_place","win_payoff","place_payoff","is_exact","is_quinella"]
    y_columns = [
        "is_win","win_payoff",
        "is_place","place_payoff",
        "is_exacta","exacta_payoff",
        "is_quinella","quinella_payoff",
    ]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 2011"
    dp = data_processor.load_from_database(db_path,x_columns,y_columns,odds_columns,where = where)
    dp.keep_separate_race_df()
    dp.under_sampling(predict_type,train_magnif = 3)
    return dp

def xgbc(features,datasets):
    train_x = datasets.get(data_processor.KEY_TRAIN_X)
    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"is_win"]
    test_x  = datasets.get(data_processor.KEY_TEST_X)
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,"is_win"]

    test_race = datasets.get(data_processor.KEY_TEST_RACE)
    test_rx = test_race["x"]
    test_rodds_win = test_race["odds"]
    test_r_win = test_race["is_win"]
    test_r_place = test_race["is_place"]
    test_rp_win = test_race["win_payoff"]
    test_rp_place = test_race["place_payoff"]
    features = train_x.columns

    xgbc = xgb.XGBClassifier(
        n_estimators = 100,
        colsample_bytree =  0.869719614599,
        learning_rate =  0.001,
        min_child_weight = 0.42206733097,
        subsample = 0.99637221573,
        max_depth = 10,
        gamma =  0.90110124545,
    )
 
    xgbc.fit(train_x,train_y)
    pred = xgbc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)

    print("")
    print("Accuracy: {0}".format(accuracy))

    importances = xgbc.feature_importances_
    evaluate.show_importance(features,importances)

    #eval(xgbc,test_rx,test_ry)

    wrapper = evaluate.ScikitWrapper(xgbc)
    print("")
    print("[*] top 1 k")
    win_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    print("[*] top 2 k")
    win_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_win,test_rp_win,n = 2)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_place,test_rp_place,n = 2)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    print("[*] top 3 k")
    win_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_win,test_rp_win,n = 3)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(wrapper,test_rx,test_r_place,test_rp_place,n = 3)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    print("")
    print("[*] top 1 k")
    win_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_win,test_rp_win,test_rodds_win)
    print("[win] buy : {0},  accuracy : {1}, payoff : {2}".format(*win_eval))
    place_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_place,test_rp_place,test_rodds_win)
    print("[place] buy : {0},  accuracy : {1}, payoff : {2}".format(*place_eval))

    print("[*] top 2 k")
    win_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_win,test_rp_win,test_rodds_win,n=2)
    print("[win] buy : {0},  accuracy : {1}, payoff : {2}".format(*win_eval))
    place_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_place,test_rp_place,test_rodds_win,n = 2)
    print("[place] buy : {0},  accuracy : {1}, payoff : {2}".format(*place_eval))

    print("[*] top 3 k")
    win_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_win,test_rp_win,test_rodds_win, n = 3)
    print("[win] buy : {0},  accuracy : {1}, payoff : {2}".format(*win_eval))
    place_eval  = evaluate.top_n_k_remove_first(wrapper,test_rx,test_r_place,test_rp_place,test_rodds_win, n=3)
    print("[place] buy : {0},  accuracy : {1}, payoff : {2}".format(*place_eval))



    report = classification_report(test_y,pred)
    print(report)

def xgbc_wigh_gridsearch(features,datasets):
    train_x = datasets.get(data_processor.KEY_TRAIN_X)
    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"is_win"]
    test_x  = datasets.get(data_processor.KEY_TEST_X)
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,"is_win"]

    test_rx = datasets.get(data_processor.KEY_TEST_RACE_X)
    test_ry = datasets.get(data_processor.KEY_TEST_RACE_Y)
    test_r_win = test_ry.loc[:,:,"is_win"]
    test_r_place = test_ry.loc[:,:,"is_place"]
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

    eval(xgbc,test_rx,test_ry)
    #win_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_win,test_rp_win)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
