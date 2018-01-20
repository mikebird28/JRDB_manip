# -*- coding:utf-8 -*-

from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,GaussianNoise,Input,Add,Conv2D,Concatenate,SpatialDropout1D
from keras.layers.core import Flatten,Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
import xgboost as xgb
import keras.backend as K
import keras.optimizers
import argparse
import numpy as np
import pandas as pd
import sqlite3
import dataset2
import util

CACHE_PATH = "./cache/emb_classify"
MODEL_PATH = "./models/emb_classify"
pd.options.display.max_rows = 1000
past_n = 3
predict_type = "is_win"

def main(use_cache = False):
    predict_type = "is_win"
    config = util.get_config("config/emb_config2.json")
    db_path = "db/output_v15.db"
    db_con = sqlite3.connect(db_path)

    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(config.features,db_con, datasets)


def generate_dataset(predict_type,db_con,config):
    print(">> loading dataset")
    main_features = config.features
    #additional_features = ["linfo_win_odds","linfo_place_odds"]
    #load_features = main_features + additional_features

    categorical_dic = dataset2.nominal_columns(db_con)
    where = "info_year > 08 and info_year < 90"
    x,y = dataset2.load_dataset(db_con,main_features,["is_win","win_payoff","is_place","place_payoff"],where = where)

    con = pd.concat([x,y],axis = 1)
    null_rate = x.isnull().sum(axis = 1)/len(x)
    con = con[null_rate == 0.0]
    x = con.loc[:,x.columns]
    y = con.loc[:,y.columns]

    main_features = sorted(x.columns.values.tolist())
    main_features_dropped = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    #mean = train_x.mean(numeric_only = True)
    #std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    mx = train_x.max()
    mn = train_x.min()
    train_x = dataset2.standardize(train_x, mx = mx, mn = mn, remove = categorical_dic)
    #train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = categorical_dic)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.standardize(test_x, mx = mx, mn = mn, remove = categorical_dic)
    #test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = categorical_dic)

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
    }
    return datasets

def dnn(features,db_con,datasets):
    train_x = datasets["train_x"]
    train_y = datasets["train_y"][predict_type].as_matrix()
    test_x  = datasets["test_x"]
    test_y  = datasets["test_y"][predict_type].as_matrix()
 
    sorted_columns = sorted(datasets["train_x"].columns.values.tolist())
    categorical_dic = dataset2.nominal_columns(db_con)

    model = create_model(sorted_columns,categorical_dic)

    train_x = [train_x.loc[:,col].as_matrix() for col in sorted_columns]
    test_x = [test_x.loc[:,col].as_matrix() for col in sorted_columns]

    try:
        model.fit(train_x,train_y,epochs = 100,batch_size = 8192,validation_data = (test_x,test_y))
    except KeyboardInterrupt:
        print("Start xgboostign")
        internal_layer = Model(inputs = model.input,outputs = model.get_layer("embbeding").output)
        i_pred = internal_layer.predict(train_x)

        xgbc = xgb.XGBClassifier(
            n_estimators = 100,
            colsample_bytree =  0.5,
            gamma = 1.0,
            learning_rate = 0.07,
            max_depth = 3,
            min_child_weight = 2.0,
            subsample = 1.0
        )
        xgbc.fit(i_pred,train_y)

        test_pred = internal_layer.predict(test_x)
        test_pred = xgbc.predict(test_pred)
        report = classification_report(test_y,test_pred)
        print(report)
    """
    for i in range(1000):
        print(i)
        model.fit(train_x,train_y,epochs = 1,batch_size = 8192)
        score = model.evaluate(test_x,test_y,verbose = 0)

        print("")
        print("test loss : {0}".format(score[0]))
        print("test acc : {0}".format(score[1]))
        save_model(model,MODEL_PATH)
    """


def create_model(sorted_columns,categorical_dic,activation = "relu",dropout = 0.3,hidden_1 = 128,hidden_2 =250,hidden_3 = 80):
    inputs = []
    flatten_layers = []

    for col in sorted_columns:
        if col in categorical_dic.keys():
            x = Input(shape = (1,),dtype = "int32")
            inputs.append(x)
            inp_dim = categorical_dic[col]+1
            out_dim = max(inp_dim//10,1)
            x = Lambda(lambda a : K.clip(a,0,inp_dim))(x)
            x = Embedding(inp_dim,out_dim,input_length = 1)(x)
            x = SpatialDropout1D(0.5)(x)
            x = Flatten()(x)
        else:
            x = Input(shape = (1,))
            inputs.append(x)
        flatten_layers.append(x)

    x = Concatenate()(flatten_layers)
    x = GaussianNoise(0.001)(x)

    x = Dense(units=hidden_1,name = "embbeding")(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units=hidden_1)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)


    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.01,epsilon = 1e-3)
    model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model 

def save_model(model,path):
    print("Save model")
    model.save(path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
