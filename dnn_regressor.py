#-*-coding:utf-8-*-
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1,l2
import argparse
import numpy as np
import pandas as pd
import sqlite3
import feature
import dataset2
import util
import evaluate
import course2vec
import place2vec

CACHE_PATH = "./cache/dnn_classify"

def main(use_cache = False):
    predict_type = "is_win"
    config = util.get_config("config/config.json")
    db_path = "db/output_v7.db"
    db_path = "db/output_v8.db"
    db_con = sqlite3.connect(db_path)
 
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(config.features_vector,datasets)
    #dnn_wigh_bayessearch(config.features,datasets)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)


def generate_dataset(predict_type,db_con,config):

    print(">> loading dataset")
    features = config.features_vector 
    x,y = dataset2.load_dataset(db_con,
        features+["info_race_course_code","pre1_race_course_code","pre2_race_course_code","pre3_race_course_code"],
        ["is_win","win_payoff","is_place","place_payoff","pre0_finishing_time"])
    #x,y = dataset2.load_dataset(db_con,
    #    features+["info_race_course_code"],
    #    ["is_win","win_payoff","is_place","place_payoff"])
    print(y["pre0_finishing_time"])



    p2v_0 = place2vec.get_vector(x["info_race_course_code"].as_matrix(),prefix = "pre0")
    x = x.drop("info_race_course_code",axis = 1)

    p2v_1 = place2vec.get_vector(x["pre1_race_course_code"].as_matrix(),prefix = "pre1")
    x = x.drop("pre1_race_course_code",axis = 1)

    p2v_2 = place2vec.get_vector(x["pre2_race_course_code"].as_matrix(),prefix = "pre2")
    x = x.drop("pre2_race_course_code",axis = 1)

    p2v_3 = place2vec.get_vector(x["pre3_race_course_code"].as_matrix(),prefix = "pre3")
    x = x.drop("pre3_race_course_code",axis = 1)

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    features = sorted(x.columns.drop("info_race_id").values.tolist())
    x = concat(x,p2v_0)
    x = concat(x,p2v_1)
    x = concat(x,p2v_2)
    x = concat(x,p2v_3)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    del x
    del y

    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    c2v_x = train_x.loc[:,features]
    c2v_df = course2vec.get_vector(c2v_x)
    train_x = concat(train_x,c2v_df)

    print(">> under sampling train dataset")
    train_x.reset_index(inplace = True,drop = True)
    train_y.reset_index(inplace = True,drop = True)
    train_x,train_y = dataset2.under_sampling(train_x,train_y)
    train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)


    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    c2v_x = test_x.loc[:,features]
    c2v_df = course2vec.get_vector(c2v_x)
    test_x = concat(test_x,c2v_df)

    print(">> under sampling test dataset")
    test_rx,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_x,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )
    test_x,test_y = dataset2.under_sampling(test_x,test_y,key = predict_type)
    test_x,test_y = dataset2.for_use(test_x,test_y,predict_type)

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
        "test_rx"      : test_rx,
        "test_ry"      : test_ry,
        "test_r_win"   : test_r_win,
        "test_r_place" : test_r_place,
        "test_rp_win"  : test_rp_win,
        "test_rp_place": test_rp_place
    }
    return datasets

def create_model(activation = "relu",dropout = 0.3,hidden_1 = 102,hidden_2 = 53,hidden_3 = 135):
    #Best Paramater of 2 hidden layer : h1 = 50, h2  = 250, dropout = 0.38
    #Best Paramater of 3 hidden layer : h1 = 138, h2  = 265, h3 = 135 dropout = 0.33 
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 262, kernel_initializer = "he_normal",activity_regularizer = l2(0.0)))
    nn.add(Activation(activation))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_2, kernel_initializer = "he_normal",activity_regularizer = l2(0.0)))
    nn.add(Activation(activation))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout))

    nn.add(Dense(units=1))
    nn.add(Activation('sigmoid'))
    nn.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return nn

def dnn(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
 
    #model = KerasClassifier(create_model,batch_size = 300,verbose = 1)
    model = create_model()
    for i in range(10):
        print(i)
        model.fit(train_x,train_y,epochs = 5,batch_size = 300)
        score = model.evaluate(test_x,test_y,verbose = 0)

        print("")

        print("test loss : {0}".format(score[0]))
        win_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    report = classification_report(test_y,pred)
    print(report)


def dnn_wigh_gridsearch(features,datasets):
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
 
    model =  KerasClassifier(create_model,nb_epoch = 6)
    paramaters = {
        hidden_1 : [50,100],
        hidden_2 : [50],
        dropout  : [1.0],
    }

    cv = GridSearchCV(model,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    win_eval  = evaluate.top_n_k(cv,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(cv,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    report = classification_report(test_y,pred)
    print(report)

    print("Paramaters")
    best_parameters = cv.best_params_
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))

def dnn_wigh_bayessearch(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
 
    model =  KerasClassifier(create_model,epochs = 6,verbose = 0)
    paramaters = {
        "hidden_1" : (10,500),
        "hidden_2" : (10,500),
        "dropout" : (0.3,0.6)
    }

    cv = BayesSearchCV(model,paramaters,cv = 3,scoring='accuracy',n_iter = 30,verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    win_eval  = evaluate.top_n_k(cv,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(cv,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    report = classification_report(test_y,pred)
    print(report)

    print("Paramaters")
    best_parameters = cv.best_params_
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))


def dataset_for_pca(x,y,mean = None,std = None):
    #x = dataset2.normalize(x,mean = mean,std = std)
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race2(x)
    return (x,y)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
