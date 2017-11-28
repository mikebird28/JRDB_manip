#-*- coding:utf-8 -*-

from skopt import BayesSearchCV
from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda,Reshape,Flatten
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as K
import keras.optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3, pickle
import dataset2, util, evaluate
import argparse


EVALUATE_INTERVAL = 10
MAX_ITERATION = 50000
BATCH_SIZE = 36
REFRESH_INTERVAL = 50 
PREDICT_TYPE = "win_payoff"
#PREDICT_TYPE = "place_payoff"
MODEL_PATH = "./models/mlp_model.h5"
MEAN_PATH = "./models/mlp_mean.pickle"
STD_PATH = "./models/mlp_std.pickle"
CACHE_PATH = "./cache/multiclass"

def main(use_cache = False):
    #predict_type = "place_payoff"
    predict_type = PREDICT_TYPE
    config = util.get_config("config/config_small.json")
    db_path = "db/output_v7.db"

    db_con = sqlite3.connect(db_path)
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    dnn(config.features,datasets)

def predict(db_con,config):
    add_col = ["info_horse_name"]
    features = config.features+add_col
    raw_x = dataset2.load_x(db_con,features)
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(raw_x,col_dic)
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
    model = load_model(MODEL_PATH)
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
    x,y = dataset2.load_dataset(db_con,features,["is_win","win_payoff","is_place","place_payoff"])
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    features = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y,test_nums = 1000)
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
    train_x,train_y = dataset2.pad_race(train_x,train_y)
    train_y["is_win"] = train_y["win_payoff"].clip(lower = 0,upper = 1)
    train_y["is_place"] = train_y["place_payoff"].clip(lower = 0,upper = 1)

    train_x,train_y = dataset2.to_race_panel(train_x,train_y)
    train_x = train_x.loc[:,:,features]
    train_y = train_y.loc[:,:,["dont_buy","place_payoff","win_payoff","is_win","is_place"]]

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> converting test dataset to race panel")
    test_x,test_y = dataset2.pad_race(test_x,test_y)
    test_y["is_win"] = test_y["win_payoff"].clip(lower = 0,upper = 1)
    test_y["is_place"] = test_y["place_payoff"].clip(lower = 0,upper = 1)

    test_x,test_y = dataset2.to_race_panel(test_x,test_y)
    test_x = test_x.loc[:,:,features]
    test_y = test_y.loc[:,:,["dont_buy","place_payoff","win_payoff","is_win","is_place"]]

    datasets = {
        "train_x" : train_x,
        "train_y" : train_y,
        "test_x"  : test_x,
        "test_y"  : test_y,
    }
    dataset2.save_cache(datasets,CACHE_PATH)
    return datasets

def dnn(features,datasets):
    print("[*] training step")
    target_y = "is_win"
    train_x = datasets["train_x"].as_matrix()
    train_y = datasets["train_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_x = datasets["test_x"].as_matrix()
    test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win = datasets["test_y"].loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T
    test_r_place = datasets["test_y"].loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    model =  create_model()
    for i in range(1000):
        model.fit(train_x,train_y,epochs = 1,batch_size = 300)
        loss,accuracy = model.evaluate(test_x,test_y,verbose = 0)
        print(loss)
        print(accuracy)
        #win_eval  = evaluate.top_n_k_keras(model,test_x,test_y,test_r_win)
        #print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        #place_eval  = evaluate.top_n_k_keras(model,test_x,test_r_place,test_r_place)
        #print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
    print("")
    print("Accuracy: {0}".format(accuracy))


def create_model(activation = "relu",dropout = 0.3,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    inputs = Input(shape = (18,132,))
    x = inputs
    x = Reshape([18,132,1],input_shape = (132*18,))(x)

    depth = 64
    x = ZeroPadding2D([[1,1],[0,0]])(x)
    x = Conv2D(depth,(3,132),padding = "valid")(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    for i in range(5):
        tmp = x
        x = ZeroPadding2D([[1,0],[0,0]])(x)
        x = Conv2D(depth,(2,1),padding = "valid")(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        x = ZeroPadding2D([[1,0],[0,0]])(x)
        x = Conv2D(depth,(2,1),padding = "valid")(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        x = Add()([x,tmp])
        x = BatchNormalization()(x)

    x = Conv2D(1,(1,1),padding = "valid")(x)
    x = Flatten()(x)
    #x = Dense(units = 18)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model

def dataset_generator(batch_size,*datasets):
    columns = []
    for d in datasets:
        columns.append(d.axes[2].tolist())
    con = pd.concat(datasets,axis = 2)
    _,i = np.unique(con.axes[2],return_index = True)
    con = con.iloc[:,:,i]

    while True:
        sample = con.sample(n = batch_size,axis = 0)
        ret = []
        for i,_ in enumerate(datasets):
            ret.append(sample.loc[:,:,columns[i]])
        yield ret

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

def huber_loss(y_true,y_pred):
    clip_value = 1.0
    x = y_true - y_pred
    condition = K.abs(x) < clip_value
    squared_loss = K.square(x)
    linear_loss = clip_value * (K.abs(x) -0.5*clip_value)
    return tf.where(condition,squared_loss,linear_loss)

def to_descrete(array):
    res = np.zeros_like(array)
    res[array.argmax(1)] = 1
    return res

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
