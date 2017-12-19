

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from keras.models import Sequential,Model,load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,Activation,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import mutual_preprocess
import argparse
import random
import keras.backend as K
import keras.optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3, pickle
import place2vec
import dataset2, util


LOOSE_VALUE = -100
DONT_BUY_VALUE = 0.0
REWARD_THREHOLD = 0
EVALUATE_INTERVAL = 10
MAX_ITERATION = 50000
BATCH_SIZE = 512
REFRESH_INTERVAL = 200 
MODEL_PATH = "./models/dqn_model2.h5"
PREDICTION_MODEL_PATH = "./models/dnn_classify"
MEAN_PATH = "./models/dqn_mean.pickle"
STD_PATH = "./models/dqn_std.pickle"
CACHE_PATH = "./cache/dqn2"

def main(use_cache = False):
    predict_type = "place_payoff"
    #predict_type = "win_payoff"
    config = util.get_config("config/config.json")
    db_path = "db/output_v12.db"

    db_con = sqlite3.connect(db_path)
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    dnn(config.features,datasets)

def generate_dataset(predict_type,db_con,config):
    print("[*] preprocessing step")
    print(">> loading dataset")
    predict_features = config.features
    x,p2v,y = mutual_preprocess.load_datasets_with_p2v(db_con,config)
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    x = concat(x,p2v)

    predict_features = sorted(x.columns.values.tolist())
    additional_features = sorted(additional_features)
    odds_features = sorted(odds_features)

    x = concat(x,x_odds)
    x = concat(x,x_race_id)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y,test_nums = 1000)
    del x,y

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    save_value(mean,MEAN_PATH)
    save_value(std,STD_PATH)

    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col+odds_features)
    train_x.loc[:,odds_features] = (train_x.loc[:,odds_features]/1000.0).clip(upper = 2.0)

    train_pred = train_x.loc[:,predict_features]
    train_odds = train_x.loc[:,odds_features]
    train_race_id = train_x.loc[:,"info_race_id"]

    print(">> pre-prediction of train dataset")
    prediction_model = load_model(PREDICTION_MODEL_PATH)
    train_pred = prediction_model.predict(train_pred.as_matrix())
    train_pred = pd.DataFrame(train_pred)
    train_pred = concat(train_pred,train_odds)
    train_pred = concat(train_pred,train_race_id)
    print(train_pred)

    concat_features = sorted(train_pred.columns.drop("info_race_id").values.tolist())

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col+odds_features)
    test_x.loc[:,odds_features] = (test_x.loc[:,odds_features]/1000.0).clip(upper = 2.0)
    test_pred = test_x.loc[:,predict_features]
    test_odds = test_x.loc[:,odds_features]
    test_race_id = test_x.loc[:,"info_race_id"]

    print(">> pre-prediction of test dataset")
    test_pred = prediction_model.predict(test_pred.as_matrix())
    test_pred = pd.DataFrame(test_pred)
    test_pred = concat(test_pred,test_odds)
    test_pred = concat(test_pred,test_race_id)


    print(">> converting train dataset to race panel")
    train_x = dataset2.downcast(train_pred)
    train_y = dataset2.downcast(train_y)
    del train_pred

    train_x,train_y = dataset2.pad_race(train_x,train_y)
    train_y["win_payoff"] = train_y["win_payoff"] + LOOSE_VALUE
    train_y["place_payoff"] = train_y["place_payoff"] + LOOSE_VALUE
    train_y["dont_buy"] = pd.DataFrame(np.zeros(len(train_y.index))+DONT_BUY_VALUE,dtype = np.float32)

    print(">> converting train dataset to race panel")
    train_x,train_action = dataset2.to_race_panel(train_x,train_y)
    train_x = train_x.loc[:,:,concat_features]
    train_action = train_action.loc[:,:,["dont_buy",predict_type]]

    print(">> converting test dataset to race panel")
    test_x = dataset2.downcast(test_pred)
    test_y = dataset2.downcast(test_y)
    del test_pred


    test_x,test_y = dataset2.pad_race(test_x,test_y)
    test_y["win_payoff"] = test_y["win_payoff"] + LOOSE_VALUE
    test_y["place_payoff"] = test_y["place_payoff"] + LOOSE_VALUE
    test_y["dont_buy"] = np.zeros(len(test_y.index),dtype = np.float32)+DONT_BUY_VALUE
    test_x,test_action = dataset2.to_race_panel(test_x,test_y)
    test_x = test_x.loc[:,:,concat_features]
    test_action = test_action.loc[:,:,["dont_buy",predict_type]]

    datasets = {
        "train_x"      : train_x,
        "train_action" : train_action,
        "test_x"       : test_x,
        "test_action"  : test_action,
    }
    dataset2.save_cache(datasets,CACHE_PATH)
    return datasets

def dnn(features,datasets):
    print("[*] training step")
    train_x = datasets["train_x"]
    train_action = datasets["train_action"]
    test_x  = datasets["test_x"]
    test_action  = datasets["test_action"]
    print(train_x.iloc[1,:,:])
    print(train_action.iloc[1,:,:])

    model =  create_model()
    old_model = keras.models.clone_model(model)

    #main_loop
    batch_size = BATCH_SIZE
    gene = dataset_generator(train_x,train_action,batch_size = batch_size)
    max_iteration = MAX_ITERATION

    for count in range(max_iteration):
        raw_x,raw_y = next(gene)
        x_ls = []
        y_ls = []
        #prob_threhold = max(float(300 - count),30.0)/1000
        prob_threhold = 0.01
        for i in range(len(raw_x)):
            rx = raw_x.ix[i]
            ry = raw_y.ix[i]

            idx = np.random.randint(18)
            #idx = 0
            new_x = rx.ix[idx,:]
            new_y = ry.ix[idx,:]
            reward = get_reward(old_model,others(rx,idx),others(ry,idx),action_threhold = prob_threhold)
            #reward = get_q_value(old_model,others(rx,idx),others(ry,idx),action_threhold = prob_threhold)
            new_y += reward
            new_y = clip(new_y)
            x_ls.append(new_x)
            y_ls.append(new_y)
        x = np.array(x_ls)
        y = np.array(y_ls)
        hist = model.fit(x,y,verbose = 0,epochs = 1,batch_size = BATCH_SIZE)
        if count % EVALUATE_INTERVAL == 0:
            #evaluate(count,model,raw_x,raw_y)
            evaluate(count,model,test_x,test_action)
            print(hist.history["loss"])
            save_model(model,MODEL_PATH)
        if count % REFRESH_INTERVAL == 0:
            old_model = keras.models.clone_model(model)
            pass

def create_model(activation = "relu",dropout = 0.6,hidden_1 = 30,hidden_2 = 250,hidden_3 = 80):
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 3))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_1))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=2))

    opt = keras.optimizers.Adam(lr=1e-4)
    #nn.compile(loss = "mean_squared_error",optimizer=opt,metrics=["accuracy"])
    #nn.compile(loss = "squared_hinge",optimizer=opt,metrics=["accuracy"])
    #nn.compile(loss = log_loss,optimizer=opt,metrics=["accuracy"])
    nn.compile(loss = huber_loss,optimizer=opt,metrics=["accuracy"])
    return nn

def evaluate(step,model,x,y):
    total_reward = 0
    total_buy = 0
    total_hit = 0
    for i in range(len(x)):
        rx = x.iloc[i]
        ry = y.iloc[i]
        is_win = ry.iloc[:,1].clip(lower = 0.0,upper = 1.0)
        buy = get_action(model,rx.as_matrix(),is_predict = True)[:,1]
        buy_num = buy.sum()
        is_hit = 1 if np.dot(is_win,buy) > 0 else 0
        reward = get_reward(model,rx,ry,is_predict = True)
        total_reward += reward
        total_buy += buy_num
        total_hit += is_hit
    avg_reward = total_reward/float(len(x))
    avg_buy = total_buy/float(len(x))
    avg_hit = total_hit/float(len(x))
    print("Step: {0}".format(step))
    print("Profit: {0} yen/race".format(avg_reward))
    print("Hit: {0} tickets/race".format(avg_hit))
    print("Buy : {0} tickets/race".format(avg_buy))
    print("")

def dataset_generator(x,y,batch_size = 100):
    x_col = x.axes[2].tolist()
    y_col = y.axes[2].tolist()
    #x.reset_index(inplace = True,drop = True)
    #y.reset_index(inplace = True,drop = True)
    con = pd.concat([x,y],axis = 2)

    while True:
        sample = con.sample(n = batch_size,axis = 0)
        x = sample.loc[:,:,x_col]
        y = sample.loc[:,:,y_col]
        yield (x,y)

def get_reward(model,x,y,is_predict = False,action_threhold = 0.01):
    x = x.as_matrix()
    y = y.as_matrix()
    action = get_action(model,x,is_predict = is_predict,threhold = action_threhold)
    rewards = (y*action).sum()
    return rewards

def get_action(model,x,is_predict = False,threhold = 0.01):
    pred = model.predict(x)
    action = pred.argmax(1)
    if not is_predict:
        for i in range(len(action)):
            prob = np.random.rand()
            if prob < threhold:
                action[i] = np.random.randint(2)
    action = np.eye(2)[action]
    return action

def get_q_value(model,x,is_predict = False,action_threhold = 0.01):
    x = x.as_matrix()
    pred = model.predict(x)
    q_value = pred.max(axis = 1).sum()
    return q_value

def clip(y):
    y = y/100.0
    #y = y/100.0
    #y[y<=REWARD_THREHOLD] = 0
    #y[y>REWARD_THREHOLD] = 1
    #y = y.clip(lower = 0.0,upper = 1.0)
    #y = y.clip(lower = 0.0,upp)
    #y = y.clip(lower = -2.0,upper = 10.0)
    y = y.clip(upper = 30.0)
    return y

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

def log_loss(y_true,y_pred):
    y_pred = K.clip(y_pred,1e-8,1.0)
    return -y_true*K.log(y_pred)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
