#-*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import keras.optimizers
import numpy as np
import pandas as pd
import sqlite3
import feature
import dataset2
import util
import evaluate
import pickle

USE_CACHE = False
LOOSE_VALUE = -100
DONT_BUY_VALUE = 0
REWARD_THREHOLD = 20
EVALUATE_INTERVAL = 5
MAX_ITERATION = 1500
MODEL_PATH = "./models/dqn_model.h5"
MEAN_PATH = "./models/dqn_mean.pickle"
STD_PATH = "./models/dqn_std.pickle"
CACHE_PATH = "./cache"

def main():
    predict_type = "place_payoff"
    config = util.get_config("config/config.json")
    db_path = "db/output_v7.db"

    db_con = sqlite3.connect(db_path)
    if USE_CACHE:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    dnn(config.features_light,datasets)

def predict(db_con,config):
    x = dataset2.load_x(db_con,config.features_light)
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

    x = dataset2.fillna_mean(x,"horse")
    mean = load_value(MEAN_PATH)
    std = load_value(STD_PATH)
    x = dataset2.normalize(x,mean = mean,std = std,remove = nom_col)
    x = dataset2.pad_race_x(x)

def generate_dataset(predict_type,db_con,config):
    print("[*] preprocessing step")
    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_con,config.features_light,["is_win","win_payoff","is_place","place_payoff"])
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

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
    train_y["win_payoff"] = train_y["win_payoff"].where(train_y["win_payoff"] != 0.0,LOOSE_VALUE)
    train_y["place_payoff"] = train_y["place_payoff"].where(train_y["place_payoff"] != 0.0,LOOSE_VALUE)
    train_y["dont_buy"] = np.zeros(len(train_y.index),dtype = np.float32)-DONT_BUY_VALUE

    train_x,train_action = dataset2.to_race_panel(train_x,train_y)
    train_x = train_x.drop("info_race_id",axis = 2)
    train_action = train_action.loc[:,:,["dont_buy",predict_type]]

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> converting test dataset to race panel")
    test_x,test_y = dataset2.pad_race(test_x,test_y)
    test_y["win_payoff"] = test_y["win_payoff"].where(test_y["win_payoff"] != 0.0,LOOSE_VALUE)
    test_y["place_payoff"] = test_y["place_payoff"].where(test_y["place_payoff"] != 0.0,LOOSE_VALUE)
    test_y["dont_buy"] = np.zeros(len(test_y.index),dtype = np.float32)-DONT_BUY_VALUE
    test_x,test_action = dataset2.to_race_panel(test_x,test_y)
    test_x = test_x.drop("info_race_id",axis = 2)
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

    model =  create_model()

    #main_loop
    gene = dataset_generator(train_x,train_action)
    max_iteration = MAX_ITERATION
    batch_size = 100

    for count in range(max_iteration):
        raw_x,raw_y = next(gene)
        x_ls = []
        y_ls = []
        for i in range(len(raw_x)):
            rx = raw_x.ix[i]
            ry = raw_y.ix[i]
            idx = np.random.randint(18)
            new_x = rx.ix[idx]
            new_y = ry.ix[idx]
            reward = get_reward(model,others(rx,i),others(ry,i))
            new_y += reward
            new_y = clip(new_y)
            x_ls.append(new_x)
            y_ls.append(new_y)
        x = np.array(x_ls)
        y = np.array(y_ls)
        prob_threnold = max(float(100 - count),10)/1000
        model.fit(x,y,verbose = 0,epochs = 1)
        if count % EVALUATE_INTERVAL == 0:
            evaluate(count,model,test_x,test_action)
            save_model(model,MODEL_PATH)

def create_model(activation = "relu",dropout = 0.4,hidden_1 = 80,hidden_2 = 250,hidden_3 = 80):
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 133, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_2, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    """
    nn.add(Dense(units=hidden_3, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))
    """

    nn.add(Dense(units=2))

    opt = keras.optimizers.Nadam(lr=0.001)
    #nn.compile(loss = "mean_squared_error",optimizer=opt,metrics=["accuracy"])
    nn.compile(loss = "hinge",optimizer=opt,metrics=["accuracy"])
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
    con = pd.concat([x,y],axis = 2)

    while True:
        sample = con.sample(n = batch_size,axis = 0)
        x = sample.loc[:,:,x_col]
        y = sample.loc[:,:,y_col]
        yield (x,y)

def get_reward(model,x,y,is_predict = False):
    x = x.as_matrix()
    y = y.as_matrix()
    action = get_action(model,x,is_predict = is_predict)
    rewards = (y*action).sum()
    return rewards

def get_action(model,x,is_predict = False,threhold = 0.1):
    pred = model.predict(x)
    action = pred.argmax(1)
    if is_predict:
        for i in range(len(action)):
            prob = np.random.rand()
            if prob < threhold:
                action[i] = np.random.randint(2)
    action = np.eye(2)[action]
    return action

def clip(y):
    y[y<=REWARD_THREHOLD] = -1
    y[y>REWARD_THREHOLD] = 1
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
        v = pickle.load(path)
        return v
    raise Exception ("File does not exist")

if __name__=="__main__":
    main()
