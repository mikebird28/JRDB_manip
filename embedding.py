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
import util
import data_processor

CACHE_PATH = "./cache/emb_classify"
MODEL_PATH = "./models/emb_classify"
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
        dp = generate_dataset(db_path,config)
        dp.save(CACHE_PATH)
    dnn(config.features,dp)


def generate_dataset(predict_type,db_path,config):
    x_columns = config.features
    y_columns = ["is_win","is_place","win_payoff","place_payoff"]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 11"

    dp = data_processor.load_from_database(db_path,x_columns,y_columns,odds_columns,where = where)
    dp.dummy()
    dp.fillna_mean()
    dp.normalize()
    #dp.standardize()
    dp.keep_separate_race_df()
    dp.under_sampling(key = "is_win")
    return dp

def dnn(features,db_con,datasets):
    """
    raw_train_x = datasets["train_x"]
    train_y = datasets["train_y"][predict_type].as_matrix()
    raw_test_x  = datasets["test_x"]
    test_y  = datasets["test_y"][predict_type].as_matrix()
    test_rx = datasets["test_rx"]
    test_r_win = datasets["test_r_win"]
    test_rp_win = datasets["test_rp_win"]
    test_r_place = datasets["test_r_place"]
    test_rp_place = datasets["test_rp_place"]
    """
    train_x = datasets.get(data_processor.KEY_TRAIN_X)
    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"is_win"]
    test_x  = datasets.get(data_processor.KEY_TEST_X)
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,"is_win"]

    test_rx = datasets.get(data_processor.KEY_TEST_RACE_X)
    test_ry = datasets.get(data_processor.KEY_TEST_RACE_Y)
    test_r_win = test_ry.loc[:,:,"is_win"]
    test_r_place = test_ry.loc[:,:,"is_place"]
    test_rp_win = test_ry.loc[:,:,"win_payoff"]
    test_rp_place = test_ry.loc[:,:,"place_payoff"]
    categorical_dic = datasets.categorical_dic
    sorted_columns = sorted(train_x.columns.values.tolist())

    train_y = train_y.as_matrix()

    model = create_model(sorted_columns,categorical_dic)

    train_x = [raw_train_x.loc[:,col].as_matrix() for col in sorted_columns]
    test_x = [raw_test_x.loc[:,col].as_matrix() for col in sorted_columns]
    new_test_rx = []
    try:
        for i in range(300):
            model.fit(train_x,train_y,epochs = 10,batch_size = 8192,validation_data = (test_x,test_y))
            print(top_n_k(model,test_rx,test_r_win,test_rp_win,sorted_columns))
            print(top_n_k(model,test_rx,test_r_place,test_rp_place,sorted_columns))
    except KeyboardInterrupt:
        pass
    print("\nStart xgboostign")
    internal_layer = Model(inputs = model.input,outputs = model.get_layer("embedding").output)

    xgbc = xgb.XGBClassifier(
        n_estimators = 100,
        colsample_bytree =  0.5,
        gamma = 1.0,
        learning_rate = 0.07,
        max_depth = 3,
        min_child_weight = 2.0,
        subsample = 1.0
    )
    train_pred = internal_layer.predict(train_x)
    train_pred = np.concatenate([train_pred,raw_train_x.as_matrix()],axis = 1)
    xgbc.fit(train_pred,train_y)

    test_pred = internal_layer.predict(test_x)
    test_pred = np.concatenate([test_pred,raw_test_x.as_matrix()],axis = 1)
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


def create_model(sorted_columns,categorical_dic,activation = "relu",dropout = 0.3,hidden_1 = 64,hidden_2 =250,hidden_3 = 80):
    inputs = []
    flatten_layers = []

    for col in sorted_columns:
        if col in categorical_dic.keys():
            x = Input(shape = (1,),dtype = "int32")
            inputs.append(x)
            inp_dim = categorical_dic[col]+1
            out_dim = max(inp_dim//10,1)
            #x = Lambda(lambda a : K.clip(a,0,inp_dim))(x)
            x = Embedding(inp_dim,out_dim,input_length = 1)(x)
            x = SpatialDropout1D(0.5)(x)
            x = Flatten()(x)
        else:
            x = Input(shape = (1,))
            inputs.append(x)
        flatten_layers.append(x)

    x = Concatenate()(flatten_layers)

    x = Dense(units=hidden_1)(x)
    x = Activation(activation,name = "embedding")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units=hidden_1)(x)
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

def top_n_k(model,race_x,race_y,payoff,sorted_columns,mode = "max"):
    counter = 0
    correct = 0
    rewards = 0
    for x,y,p in zip(race_x,race_y,payoff):
        #x = x.reshape([1,x.shape[0],x.shape[1]])
        x = [x.loc[:,col].as_matrix() for col in sorted_columns]
        pred = model.predict(x,verbose = 0)
        binary_pred = to_descrete(pred,mode = mode)
        b = np.array(binary_pred).ravel()
        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,b)
        ret = np.dot(p,b)
        if c > 0:
            correct +=1
            rewards += ret
        counter += 1
    return (float(correct)/counter,float(rewards)/counter)

def to_descrete(array,mode = "max"):
    res = np.zeros_like(array)
    res[array.argmax(0)] = 1
    return res

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
