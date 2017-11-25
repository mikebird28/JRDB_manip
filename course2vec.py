#-*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda,Reshape,Flatten
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
import argparse
import xgboost as xgb
import random
import keras.backend as K
import keras.optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3, pickle
import dataset2, util, evaluate, feature


BATCH_SIZE = 36
PREDICT_TYPE = "is_win"
#PREDICT_TYPE = "is_place"
MODEL_PATH = "./models/course2vec.h5"
PREDICT_MODEL_PATH = "./models/dqn_model2.h5"
MEAN_PATH = "./models/dqn_mean.pickle"
STD_PATH = "./models/dqn_std.pickle"
CACHE_PATH = "./cache/course2vec"

def main(use_cache = False):
    predict_type = PREDICT_TYPE
    config = util.get_config("config/config.json")
    db_path = "db/output_v9.db"
    db_con = sqlite3.connect(db_path)
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    print(datasets["train_x"].isnull().sum().sum())
    print(datasets["train_y"].isnull().sum().sum())
    dnn(config.features,datasets)
    #dnn_wigh_bayessearch(config.features,datasets)
    xgboost_test(datasets)

def add_vector(x):
    matrix_x = x.as_matrix()
    model = load_model(MODEL_PATH)
    vectors = pd.DataFrame(model.predict(matrix_x))
    x.reset_index(drop = True,inplace = True)
    vectors.reset_index(drop = True,inplace = True)
    x = concat(x,vectors)
    return x

def get_vector(x,nom_col):
 
    mean = load_value(MEAN_PATH)
    std = load_value(STD_PATH)
    x = dataset2.normalize(x,mean = mean,std = std,remove = nom_col)
 
    matrix_x = x.as_matrix()
    model = load_model(MODEL_PATH)
    vectors = pd.DataFrame(model.predict(matrix_x))
    return vectors

def xgboost_test(datasets):
    train_x = datasets["train_x_pred"]
    train_y = datasets["train_y_pred"]
    train_c = datasets["train_y"]

    test_x = datasets["test_x_pred"]
    test_y = datasets["test_y_pred"]
    test_c = datasets["test_y"]


    train_x = add_vector(train_x)
    test_x = add_vector(test_x)

    xgbc = xgb.XGBClassifier(
        n_estimators = 3000,
        colsample_bytree =  0.5,
        gamma = 1.0,
        learning_rate = 0.07,
        max_depth = 3,
        min_child_weight = 2.0,
        subsample = 1.0
        )
    xgbc.fit(train_x,train_y)

    pred = xgbc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    importances = xgbc.feature_importances_
    features = train_x.columns
    evaluate.show_importance(features,importances)

def predict(x):
    raw_x = dataset2.get_dummies(raw_x,col_dic)

    target_columns = []
    remove = ["info_race_id","info_horse_name","info_race_name"]
    for col in raw_x.columns:
        if col not in remove:
            target_columns.append(col)
    target_columns = sorted(target_columns)

    x = dataset2.fillna_mean(raw_x,"horse")
    mean = load_value(MEAN_PATH)
    std = load_value(STD_PATH)
    x = dataset2.normalize(x,mean = mean,std = std,remove = nom_col+add_col)
    x = dataset2.pad_race_x(x)
    x = dataset2.to_race_panel(x)[0]

    inputs = x.loc[:,:,target_columns]
    model = load_model(PREDICT_MODEL_PATH)
    actions = []
    for i in range(len(inputs)):
        rx = x.iloc[i,:,:]
        ri = inputs.iloc[i,:,:]

        a = get_action(model,ri.as_matrix(),is_predict = True)
        a = pd.DataFrame(a,columns = ["dont_buy","buy"])
        rx = pd.concat([rx,a],axis = 1)
        print(rx.loc[:,["info_horse_name","buy"]])
        print("")
        if i > 12:
            break
    actions = pd.Panel(actions)


def generate_dataset(predict_type,db_con,config):
    print("[*] preprocessing step")
    print(">> loading dataset")

    features = config.features
    target = "target"

    x,y = dataset2.load_dataset(db_con,features,["is_win","is_place","info_race_course_code","rinfo_discipline"])
    x_col = x.columns 
    y_col = y.columns
    con = concat(x,y)
    con = con[con["info_race_course_code"] != 0]
    con = con[con["rinfo_discipline"] != 0]
    x = con.loc[:,x_col]
    y = con.loc[:,y_col]

    y["info_race_course_code"] = y["info_race_course_code"] - 1
    y["rinfo_discipline"] = y["rinfo_discipline"] - 1
    y[target] = y["info_race_course_code"] * 3 + y["rinfo_discipline"]

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    #features = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")

    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y,test_nums = 1000)
    train_x = dataset2.get_dummies(train_x,col_dic)
    test_x = dataset2.get_dummies(test_x,col_dic)
    features = sorted(train_x.columns.drop("info_race_id").values.tolist())

    del x
    del y

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    save_value(mean,MEAN_PATH)
    save_value(std,STD_PATH)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> converting train dataset to race panel")
    train_x = dataset2.downcast(train_x)
    train_y = dataset2.downcast(train_y)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> converting test dataset to race panel")

    con = concat(train_x,train_y)
    con = con[con[predict_type] == 1]
    train_win_x = con.loc[:,features]
    train_win_y = con.loc[:,target]
    train_x_pred = train_x.loc[:,features]
    train_y_pred = train_y.loc[:,predict_type]
    train_x_pred,train_y_pred = dataset2.under_sampling(train_x_pred,train_y_pred,key = predict_type)

    con = concat(test_x,test_y)
    con = con[con[predict_type] == 1]
    test_win_x = con.loc[:,features]
    test_win_y = con.loc[:,target]
    test_x_pred = test_x.loc[:,features]
    test_y_pred = test_y.loc[:,predict_type]
    test_x_pred,test_y_pred = dataset2.under_sampling(test_x_pred,test_y_pred,key = predict_type)

    train_win_y = dataset2.get_dummies(train_win_y,{target:32})
    test_win_y = dataset2.get_dummies(test_win_y,{target:32})

    datasets = {
        "train_x" : train_win_x,
        "train_y" : train_win_y,
        "test_x"  : test_win_x,
        "test_y"  : test_win_y,
        "train_x_pred": train_x_pred,
        "train_y_pred": train_y_pred,
        "test_x_pred": test_x_pred,
        "test_y_pred": test_y_pred,
    }

    dataset2.save_cache(datasets,CACHE_PATH)
    return datasets

def dnn(features,datasets):
    print("[*] training step")
    target_y = "is_win"
    train_x = datasets["train_x"].as_matrix()
    train_y = datasets["train_y"].as_matrix()

    test_x = datasets["test_x"].as_matrix()
    test_y = datasets["test_y"].as_matrix()

    model =  create_model()
    internal = Model(inputs = model.input,outputs = model.get_layer("internal").output)
    for i in range(100):
        model.fit(train_x,train_y,epochs = 1,batch_size = 300)
        loss,accuracy = model.evaluate(test_x,test_y,verbose = 0)
        print("")
        print(loss)
        result = internal.predict(test_x,verbose = 0)
    save_model(internal,MODEL_PATH)

def dnn_wigh_bayessearch(features,datasets):
    print("[*] training step")
    target_y = "is_win"
    train_x = datasets["train_x"].as_matrix()
    train_y = datasets["train_y"].as_matrix()

    test_x = datasets["test_x"].as_matrix()
    test_y = datasets["test_y"].as_matrix()


    model =  KerasClassifier(create_model,epochs = 10,batch_size = 300,verbose = 0)
    paramaters = {
        "hidden_1" : (10,100),
        "dropout" : (0.3,1.0)
    }

    cv = BayesSearchCV(model,paramaters,cv = 3,scoring='f1_macro',n_iter = 1,verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    print("Paramaters")
    best_parameters = cv.best_params_
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))


def create_model(activation = "relu",dropout = 0.2,hidden_1 = 20):
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 216))
    nn.add(Activation(activation))
    nn.add(BatchNormalization(name = "internal"))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=33))
    nn.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.1)
    nn.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    return nn

def others(df,idx):
    df = df[~df.index.isin([idx])]
    return df

def save_model(model,path):
    print("Save model")
    model.save(path)

def save_value(v,path):
    with open(path,"wb") as fp:
        pickle.dump(v,fp)

def load_value(path):
    with open(path,"rb") as fp:
        v = pickle.load(fp)
        return v
    raise Exception ("File does not exist")

def top_n_k(model,x,y,payoff):
    pred = model.predict(x,verbose = 0)
    binary_pred = to_descrete(pred)
    print(pred[0])
    print(binary_pred[0])
    c = y*binary_pred
    correct = np.sum(c)
    ret = payoff.T*binary_pred
    reward = np.sum(ret)

    return correct,reward

def to_descrete(array):
    res = np.zeros_like(array)
    res[array.argmax(1)] = 1
    return res

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
