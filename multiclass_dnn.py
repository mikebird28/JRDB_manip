# -*- coding:utf-8 -*-

from skopt import BayesSearchCV
from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
import keras.optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3, pickle
import dataset2, util, evaluate
import argparse
import place2vec
import course2vec
import field_fitness


EVALUATE_INTERVAL = 10
MAX_ITERATION = 50000
BATCH_SIZE = 36
REFRESH_INTERVAL = 50 
PREDICT_TYPE = "win_payoff"
#PREDICT_TYPE = "place_payoff"
MODEL_PATH = "./models/mlp_model3.h5"
MEAN_PATH = "./models/mlp_mean3.pickle"
STD_PATH = "./models/mlp_std3.pickle"
CACHE_PATH = "./cache/multiclass3"

pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
 
def main(use_cache = False):
    #predict_type = "place_payoff"
    predict_type = PREDICT_TYPE
    config = util.get_config("config/config.json")
    db_path = "db/output_v11.db"

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
    x,y = dataset2.load_dataset(db_con,
        features+["info_race_course_code","rinfo_discipline","rinfo_distance",
                  "pre1_race_course_code","pre1_discipline","pre1_distance",
                  "pre2_race_course_code","pre2_discipline","pre2_distance",
                  "pre3_race_course_code","pre3_discipline","pre3_distance"],
        ["is_win","win_payoff","is_place","place_payoff"])
    con = concat(x,y)
    x_col = x.columns
    y_col = y.columns
    con = con[con["info_race_course_code"] != 0]
    con = con[con["rinfo_discipline"] != 0]
    con = con[con["pre1_discipline"] != 0]
    con = con[con["pre1_distance"] != 0]
    con = con[con["pre1_race_course_code"] != 0]
    con = con[con["pre2_discipline"] != 0]
    con = con[con["pre2_distance"] != 0]
    con = con[con["pre2_race_course_code"] != 0]
    con = con[con["pre3_distance"] != 0]
    con = con[con["pre3_race_course_code"] != 0]
    con.reset_index(drop = True,inplace = True)
    x = con.loc[:,x_col]
    y = con.loc[:,y_col]
    del con

    p2v_0 = place2vec.get_vector(x["rinfo_discipline"],x["info_race_course_code"],x["rinfo_distance"],prefix = "pre0")
    x = x.drop("info_race_course_code",axis = 1)
    x = x.drop("rinfo_discipline",axis = 1)
    x = x.drop("rinfo_distance",axis = 1)

    p2v_1 = place2vec.get_vector(x["pre1_discipline"],x["pre1_race_course_code"],x["pre1_distance"],prefix = "pre1")
    x = x.drop("pre1_race_course_code",axis = 1)
    x = x.drop("pre1_discipline",axis = 1)
    x = x.drop("pre1_distance",axis = 1)

    p2v_2 = place2vec.get_vector(x["pre2_discipline"],x["pre2_race_course_code"],x["pre2_distance"],prefix = "pre2")
    x = x.drop("pre2_race_course_code",axis = 1)
    x = x.drop("pre2_discipline",axis = 1)
    x = x.drop("pre2_distance",axis = 1)

    p2v_3 = place2vec.get_vector(x["pre3_discipline"],x["pre3_race_course_code"],x["pre3_distance"],prefix = "pre3")
    x = x.drop("pre3_race_course_code",axis = 1)
    x = x.drop("pre3_discipline",axis = 1)
    x = x.drop("pre3_distance",axis = 1)

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    raw_features = sorted(x.columns.drop("info_race_id").values.tolist())

    x = concat(x,p2v_0)
    x = concat(x,p2v_1)
    x = concat(x,p2v_2)
    x = concat(x,p2v_3)
    x = dataset2.downcast(x)
    del p2v_0
    del p2v_1
    del p2v_2
    del p2v_3
    features = sorted(x.columns.drop("info_race_id").values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y,test_nums = 1000)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)
    del x
    del y

    train_x = dataset2.fillna_mean(train_x,"horse")

    c2v_x = train_x.loc[:,raw_features]
    c2v_df = course2vec.get_vector(c2v_x,nom_col)
    train_x = concat(train_x,c2v_df)
    train_y.reset_index(inplace = True,drop = True)
    del c2v_x
    del c2v_df

    ff_x = train_x.loc[:,raw_features]
    ff_df = field_fitness.get_vector(ff_x,nom_col)
    train_x = concat(train_x,ff_df)
    test_y.reset_index(drop = True,inplace = True)
    del ff_x
    del ff_df

    print(">> filling none value of train dataset")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    save_value(mean,MEAN_PATH)
    save_value(std,STD_PATH)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> converting train dataset to race panel")
    features = sorted(train_x.columns.drop("info_race_id").values.tolist())
    train_x = dataset2.downcast(train_x)
    train_y = dataset2.downcast(train_y)
    train_x,train_y = dataset2.pad_race(train_x,train_y)
    train_x = dataset2.downcast(train_x)
    train_y = dataset2.downcast(train_y)
    train_y["is_win"] = train_y["win_payoff"].clip(lower = 0,upper = 1)
    train_y["is_place"] = train_y["place_payoff"].clip(lower = 0,upper = 1)

    train_x,train_y = dataset2.to_race_panel(train_x,train_y)

    train_x1 = train_x.ix[:20000,:,features]
    train_x2 = train_x.ix[20000:,:,features]
    del train_x
    train_y = train_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]

    test_x = dataset2.fillna_mean(test_x,"horse")

    c2v_x = test_x.loc[:,raw_features]
    c2v_df = course2vec.get_vector(c2v_x,nom_col)
    test_x = concat(test_x,c2v_df)
    test_y.reset_index(drop = True,inplace = True)
    del c2v_x
    del c2v_df

    ff_x = test_x.loc[:,raw_features]
    ff_df = field_fitness.get_vector(ff_x,nom_col)
    test_x = concat(test_x,ff_df)
    test_y.reset_index(drop = True,inplace = True)
    del ff_x
    del ff_df


    print(">> filling none value of test dataset")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> converting test dataset to race panel")
    test_x,test_y = dataset2.pad_race(test_x,test_y)
    test_y["is_win"] = test_y["win_payoff"].clip(lower = 0,upper = 1)
    test_y["is_place"] = test_y["place_payoff"].clip(lower = 0,upper = 1)

    test_x,test_y = dataset2.to_race_panel(test_x,test_y)
    test_x = test_x.loc[:,:,features]
    test_y = test_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]

    datasets = {
        "train_x1" : train_x1,
        "train_x2" : train_x2,
        "train_y" : train_y,
        "test_x"  : test_x,
        "test_y"  : test_y,
    }
    dataset2.save_cache(datasets,CACHE_PATH)
    return datasets

def dnn(features,datasets):
    print("[*] training step")
    target_y = "is_win"
    #train_x = pd.concat([datasets["train_x1"],datasets["train_x2"]],axis = 0).as_matrix()
    train_x = pd.concat([datasets["train_x1"],datasets["train_x2"]],axis = 0)
    train_x = train_x.as_matrix()
    train_y = datasets["train_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_x = datasets["test_x"].as_matrix()
    test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win = datasets["test_y"].loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T
    test_r_place = datasets["test_y"].loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    model =  create_model()
    for i in range(1000):
        print("Epoch : {0}".format(i))
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


def create_model(activation = "relu",dropout = 0.8,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    feature_size = 356
    l2_coef = 0.0
    inputs = Input(shape = (18,feature_size,))
    x = inputs
    x = GaussianNoise(0.01)(x)
    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)

    depth = 64 
    x = Conv2D(depth,(1,feature_size),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    race_depth = 8
    tmp = Permute((2,3,1),input_shape = (1,18,depth))(x)
    tmp = Conv2D(race_depth,(1,depth),padding = "valid",kernel_regularizer = l2(l2_coef))(tmp)
    tmp = Activation(activation)(tmp)
    tmp = Conv2D(race_depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(tmp)
    tmp = Dropout(0.8)(tmp)
    tmp = Lambda(lambda x : K.tile(x,(1,18,1,1)))(tmp)
    x = Concatenate(axis = 3)([x,tmp])

    depth = depth + race_depth
    for i in range(16):
        res = x
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)
        x = Conv2D(depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout)(x)
        x = Conv2D(depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
        x = Add()([x,res])

    x = Conv2D(1,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
    x = Flatten()(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.001)
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

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
