# -*- coding:utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,GaussianNoise,Input,Add,Conv2D,Concatenate,LocallyConnected2D
from keras.layers.core import Flatten, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.regularizers import l1,l2
import keras.optimizers
import argparse
import numpy as np
import pandas as pd
import sqlite3
import dataset2
import util
import evaluate,course2vec,place2vec,mutual_preprocess

CACHE_PATH = "./cache/dnn_classify"
MODEL_PATH = "./models/dnn_classify"
pd.options.display.max_rows = 1000
past_n = 3

def main(use_cache = False):
    predict_type = "is_win"
    config = util.get_config("config/xgbc_config.json")
    db_path = "db/output_v15.db"
    db_con = sqlite3.connect(db_path)

    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(config.features, datasets)

def generate_dataset(predict_type,db_con,config):
    print(">> loading dataset")
    main_features = config.features
    #additional_features = ["linfo_win_odds","linfo_place_odds"]
    additional_features = []
    load_features = main_features + additional_features
    x,p2v,y = mutual_preprocess.load_datasets_with_p2v(db_con,load_features)
    x = concat(x,p2v)

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    x = dataset2.downcast(x)

    main_features = sorted(x.columns.values.tolist())
    main_features_dropped = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling train dataset")
    train_x.reset_index(inplace = True,drop = True)
    train_y.reset_index(inplace = True,drop = True)
    train_x,train_y = dataset2.under_sampling(train_x,train_y,key = predict_type,magnif = 3)

    print(">> under sampling train dataset")
    test_x.reset_index(inplace = True,drop = True)
    test_y.reset_index(inplace = True,drop = True)
    test_x,test_y = dataset2.under_sampling(test_x,test_y,key = predict_type)

    train_x_c,train_x_p = separate_cuurent_and_past(train_x,main_features_dropped,past_n)
    test_x_c,test_x_p = separate_cuurent_and_past(test_x,main_features_dropped,past_n)

    datasets = {
        "train_x_c"      : train_x_c,
        "train_x_p"      : train_x_p,
        "train_y"        : train_y,
        "test_x_c"       : test_x_c,
        "test_x_p"       : test_x_p,
        "test_y"         : test_y,
    }
    return datasets


def dnn(features,datasets):
    train_x_c = datasets["train_x_c"].as_matrix()
    print(datasets["train_x_p"].axes[1])
    print(datasets["train_x_p"].axes[2])
    train_x_p = datasets["train_x_p"].as_matrix()
    train_y = datasets["train_y"].loc[:,"is_win"].as_matrix()
    test_x_c  = datasets["test_x_c"].as_matrix()
    test_x_p  = datasets["test_x_p"].as_matrix()
    test_y  = datasets["test_y"].loc[:,"is_win"].as_matrix()
    """
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    #test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
    """
 
    model = create_model()
    model.fit([train_x_c,train_x_p],train_y,epochs = 100,batch_size = 4096,validation_data = ([test_x_c,test_x_p],test_y))
    """
    for i in range(1000):
        print(i)
        model.fit([train_x_c,train_x_p],train_y,epochs = 1,batch_size = 4096)
        score = model.evaluate([test_x_c,test_x_p],test_y,verbose = 0)

        print("")
        print("test loss : {0}".format(score[0]))
        print("test acc : {0}".format(score[1]))
        win_eval  = top_n_k(model,test_rx,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = top_n_k(model,test_rx,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
        save_model(model,MODEL_PATH)
    """

def create_model(activation = "relu",dropout = 0.5,hidden_1 = 80,hidden_2 =250,hidden_3 = 80):
    #def create_model(activation = "relu",dropout = 0.3,hidden_1 = 200,hidden_2 =250,hidden_3 = 135):
    #Best Paramater of 2 hidden layer : h1 = 50, h2  = 250, dropout = 0.38
    #Best Paramater of 3 hidden layer : h1 = 138, h2  = 265, h3 = 135 dropout = 0.33 
    l2_lambda = 0.000
    hidden_1 = hidden_3 = 60

    past_inputs = Input(shape = (3,33,))
    px = GaussianNoise(0.001)(past_inputs)

    px = TimeDistributed(Dense(units = 40,kernel_regularizer = l1(l2_lambda)))(px)
    px = TimeDistributed(Activation(activation))(px)
    px = TimeDistributed(BatchNormalization())(px)
    px = TimeDistributed(Dropout(dropout))(px)

    px = LSTM(units = 40,go_backwards = True)(px)
    """
    px = Dense(units=40, kernel_regularizer = l1(l2_lambda))(px)
    px = Activation(activation)(px)
    px = BatchNormalization()(px)
    px = Dropout(dropout)(px)
    """

    current_inputs = Input(shape = (284,))
    x = GaussianNoise(0.001)(current_inputs)
    x = Dense(units=40, kernel_regularizer = l1(l2_lambda))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Concatenate()([x,px])
    x = Dense(units=hidden_1, kernel_regularizer = l2(l2_lambda))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    depth = 2
    for i in range(depth):
        residual = x
        x = Dense(units=hidden_3,kernel_regularizer = l2(l2_lambda))(x)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)

        x = Dense(units=hidden_3,kernel_regularizer = l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Add()([x,residual])
        x = Dropout(dropout)(x)
        x = Activation(activation)(x)

    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs = [current_inputs,past_inputs],outputs = x)
    opt = keras.optimizers.Adam(lr=0.01,epsilon = 1e-2)
    model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model 

def save_model(model,path):
    print("Save model")
    model.save(path)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

def top_n_k(model,race_x,race_y,payoff,mode = "max"):
    counter = 0
    correct = 0
    rewards = 0
    for x,y,p in zip(race_x,race_y,payoff):
        pred = model.predict(x,verbose = 0)
        binary_pred = evaluate.to_descrete(pred,mode = mode)
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

def separate_cuurent_and_past(df,features,past_n):
    current_col = sorted([col for col in features if not col.startswith("pre")])
    pre_i_col = []
    for i in range(past_n):
        prefix = "pre{0}_".format(i+1)
        ls = [col for col in features if col.startswith(prefix)]
        ls = sorted(ls)
        pre_i_col.append(ls)

    df_current = df.loc[:,current_col]
    df_past_dic = {}

    for i in range(len(pre_i_col)):
        prefix = "pre{0}_".format(i+1)
        pre_columns = pre_i_col[i]
        past_df = df.loc[:,pre_columns]
        new_columns = map(lambda x:x.replace(prefix,""),pre_columns)
        past_df.columns = new_columns
        print(past_df.columns)
        df_past_dic[str(i)] = past_df
    df_past = pd.Panel(df_past_dic)
    df_past = df_past.swapaxes(0,1)
    return (df_current,df_past)

def separate_exinfo(x):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
