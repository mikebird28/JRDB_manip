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
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
import random
import keras.backend as K
import keras.optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3, pickle
import dataset2, util, evaluate, feature


USE_CACHE = True
LOOSE_VALUE = -100
DONT_BUY_VALUE = -20.0
REWARD_THREHOLD = -10
EVALUATE_INTERVAL = 10
MAX_ITERATION = 50000
BATCH_SIZE = 36
REFRESH_INTERVAL = 50 
PREDICT_TYPE = "is_win"
#PREDICT_TYPE = "place_payoff"
MODEL_PATH = "./models/course2vec.h5"
PREDICT_MODEL_PATH = "./models/dqn_model2.h5"
MEAN_PATH = "./models/dqn_mean.pickle"
STD_PATH = "./models/dqn_std.pickle"
CACHE_PATH = "./cache/course2vec"

def main():
    predict_type = PREDICT_TYPE
    config = util.get_config("config/config.json")
    db_path = "db/output_v7.db"

    db_con = sqlite3.connect(db_path)
    if USE_CACHE:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    dnn(config.features,datasets)
    #dnn2(config.features_light,datasets)
    xgboost_test(datasets)

def add_vector(x):
    matrix_x = x.as_matrix()
    model = load_model(MODEL_PATH)
    vectors = pd.DataFrame(model.predict(matrix_x))
    x.reset_index(drop = True,inplace = True)
    vectors.reset_index(drop = True,inplace = True)
    x = pd.concat([x,vectors],axis = 1)
    return x

def get_vector(x):
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
    """
    model = load_model(MODEL_PATH)

    vectors = model.predict(train_x.as_matrix())
    vectors = pd.DataFrame(vectors)
    train_x.reset_index(drop = True,inplace = True)
    train_c.reset_index(drop = True,inplace = True)
    vectors.reset_index(drop = True,inplace = True)
    train_x = pd.concat([train_x,vectors],axis = 1)
    #train_x = pd.concat([train_x,vectors,train_c],axis = 1)
    """

    """
    vectors = model.predict(test_x.as_matrix())
    vectors = pd.DataFrame(vectors)
    test_x.reset_index(drop = True,inplace = True)
    test_c.reset_index(drop = True,inplace = True)
    vectors.reset_index(drop = True,inplace = True)
    test_x = pd.concat([test_x,vectors],axis = 1)
    """
    #test_x = pd.concat([test_x,vectors,test_c],axis = 1)
    xgbc = xgb.XGBClassifier(
        n_estimators = 1000,
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

    #win_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_win,test_rp_win)
    #print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    #place_eval  = evaluate.top_n_k(xgbc,test_rx,test_r_place,test_rp_place)
    #print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    report = classification_report(test_y,pred)
    #print(report)

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
    #features = config.features_vector
    features = config.features
    x,y = dataset2.load_dataset(db_con,features,["is_win","is_place","info_race_course_code"])

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    features = sorted(x.columns.drop("info_race_id").values.tolist())

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

    con = pd.concat([train_x,train_y],axis = 1)
    con = con[con["is_win"] == 1]
    #con = con[con["is_place"] == 1]
    train_win_x = con.loc[:,features]
    train_win_y = con.loc[:,"info_race_course_code"]
    train_x_pred = train_x.loc[:,features]
    train_y_pred = train_y.loc[:,predict_type]
    train_x_pred,train_y_pred = dataset2.under_sampling(train_x_pred,train_y_pred)

    con = pd.concat([test_x,test_y],axis = 1)
    #con = con[con["is_place"] == 1]
    con = con[con["is_win"] == 1]
    test_win_x = con.loc[:,features]
    test_win_y = con.loc[:,"info_race_course_code"]
    test_x_pred = test_x.loc[:,features]
    test_y_pred = test_y.loc[:,predict_type]
    test_x_pred,test_y_pred = dataset2.under_sampling(test_x_pred,test_y_pred)

    train_win_y = dataset2.get_dummies(train_win_y,col_dic)
    test_win_y = dataset2.get_dummies(test_win_y,col_dic)

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
    for i in range(15):
        model.fit(train_x,train_y,epochs = 1,batch_size = 300)
        loss,accuracy = model.evaluate(test_x,test_y,verbose = 0)
        print("")
        print(loss)
        result = internal.predict(test_x,verbose = 0)
    save_model(internal,MODEL_PATH)

def dnn2(features,datasets):
    print("[*] training step")
    target_y = "is_win"
    train_x = datasets["train_x"].as_matrix()
    train_y = datasets["train_y"].as_matrix()

    test_x = datasets["test_x"].as_matrix()
    test_y = datasets["test_y"].as_matrix()

    model =  create_model2()
    internal = Model(inputs = model.input,outputs = model.get_layer("internal").output)
    for i in range(40):
        model.fit(train_y,train_x,epochs = 1,batch_size = 300)
        loss,accuracy = model.evaluate(test_y,test_x,verbose = 0)
        print("")
        print(loss)
    #save_model(internal,MODEL_PATH)

def create_model(activation = "relu",dropout = 0.3,hidden_1 = 120,hidden_2 = 120,hidden_3 = 120):
    inputs = Input(shape = (132,))
    x = inputs

    """
    x = Dense(units = 30)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    """

    x = Dense(units = 15,name = "internal")(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units = 11)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.1)
    model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model

def create_model2(activation = "relu",dropout = 0.3,hidden_1 = 120,hidden_2 = 120,hidden_3 = 120):
    inputs = Input(shape = (11,))
    x = inputs

    x = Dense(units = 200, name = "internal")(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    outputs = Dense(units = 134)(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.1)
    model.compile(loss = "mean_squared_error",optimizer=opt,metrics=["accuracy"])
    return model



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

if __name__=="__main__":
    main()
