#-*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasRegressor
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
#PREDICT_TYPE = "win_payoff"
PREDICT_TYPE = "place_payoff"
MODEL_PATH = "./models/dqn_model2.h5"
PREDICT_MODEL_PATH = "./models/dqn_model2.h5"
MEAN_PATH = "./models/dqn_mean.pickle"
STD_PATH = "./models/dqn_std.pickle"
CACHE_PATH = "./cache/dueling_dqn"

def main():
    #predict_type = "place_payoff"
    predict_type = PREDICT_TYPE
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
    add_col = ["info_horse_name"]
    features = config.features_light+add_col
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

    print(actions)

def generate_dataset(predict_type,db_con,config):
    print("[*] preprocessing step")
    print(">> loading dataset")
    features = config.features_light
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
    train_y["win_payoff"] = train_y["win_payoff"] + LOOSE_VALUE
    train_y["place_payoff"] = train_y["place_payoff"] + LOOSE_VALUE
    train_y["dont_buy"] = pd.DataFrame(np.zeros(len(train_y.index))+DONT_BUY_VALUE,dtype = np.float32)
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
    test_y["win_payoff"] = test_y["win_payoff"] + LOOSE_VALUE
    test_y["place_payoff"] = test_y["place_payoff"] + LOOSE_VALUE
    test_y["dont_buy"] = np.zeros(len(test_y.index),dtype = np.float32)+DONT_BUY_VALUE
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
    target_y = "is_place"
    train_x = datasets["train_x"]
    train_y = datasets["train_y"].loc[:,:,[target_y]]
    train_action = datasets["train_y"].loc[:,:,["dont_buy",PREDICT_TYPE]]
    #train_action.loc[:,:,"dont_buy"] = -10

    test_x = datasets["test_x"]
    test_y = datasets["test_y"].loc[:,:,target_y]
    test_action  = datasets["test_y"].loc[:,:,["dont_buy",PREDICT_TYPE]]

    rl_memory = []
    sl_memory_x = []
    sl_memory_y = []

    p_model =  create_policy_model()
    a_model = create_action_model()
    old_a_model = keras.models.clone_model(a_model)

    #main_loop
    batch_size = BATCH_SIZE
    gene = dataset_generator(batch_size,train_x,train_y,train_action)
    max_iteration = MAX_ITERATION

    total_reward = 0.0
    avg_reward = -10
 
    for count in range(max_iteration):
        raw_x,raw_y,raw_reward = next(gene)
        x_ls = []
        y_ls = []
        a_ls = []
        prob_threhold = max(float(100 - count),10.0)/1000
        rob_threhold = 0.0
        total_reward = 0
        avg_reward = -10
        for i in range(len(raw_x)):
            prob = np.random.rand()
            model_threhold = 0.1
            if prob < model_threhold:
                use_model = a_model
            else:
                use_model = p_model
            rx = raw_x.ix[i]
            ry = raw_y.ix[i]
            ra = raw_reward.ix[i]

            idx = np.random.randint(18)
            new_x = rx.ix[idx]
            new_y = ry.ix[idx]
            new_a = ra.ix[idx]
            reward = get_reward(use_model,others(rx,idx),others(ra,idx),action_threhold = prob_threhold)
            new_a += reward
            total_reward += reward
            new_a = clip(new_a)

            if(new_a.max() > avg_reward):
                if new_a.argmax(1) == "dont_buy":
                    a = [1,0]
                else:
                    a = [0,1]
                sl_memory_x.append(new_x)
                sl_memory_y.append(a)
            x_ls.append(new_x)
            y_ls.append(new_y)
            a_ls.append(new_a)
        avg_reward = total_reward/BATCH_SIZE
        total_reward = 0.0
        x = np.array(x_ls)
        y = np.array(y_ls)
        a = np.array(a_ls)
        hist = a_model.fit(x,a,verbose = 0,epochs = 1)
        if len(sl_memory_x) > 30:
            sl_memory_x = np.array(sl_memory_x)
            sl_memory_y = np.array(sl_memory_y)
            p_model.fit(sl_memory_x,sl_memory_y,verbose = 0,epochs = 1)
            sl_memory_x = []
            sl_memory_y = []
        if count % EVALUATE_INTERVAL == 0:
            evaluate(count,p_model,test_x,test_action)
            print(hist.history["loss"][0])
            print("")
            #save_model(model,MODEL_PATH)
        if count % REFRESH_INTERVAL == 0:
            old_a_model = keras.models.clone_model(a_model)

def create_policy_model(activation = "relu",dropout = 0.6,hidden_1 = 120,hidden_2 = 120,hidden_3 = 120):
    inputs_size = 134
    actions_size = 2

    inputs = Input(shape = (inputs_size,))
    x = inputs

    x = Dense(units = hidden_1)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units = hidden_2)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units = 2)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss = ["categorical_crossentropy"],optimizer=opt,metrics=["accuracy"])
    return model

def create_action_model(activation = "relu",dropout = 0.6,hidden_1 = 120,hidden_2 = 120,hidden_3 = 120):
    inputs_size = 134
    actions_size = 2

    inputs = Input(shape = (inputs_size,))
    x = inputs

    x = Dense(units = hidden_1)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units = hidden_2)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units = 2)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss = ["mean_squared_error"],optimizer=opt,metrics=["accuracy"])
    #model.compile(loss = ["categorical_crossentropy"],optimizer=opt,metrics=["accuracy"])
    return model


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


def get_q_value(model,x,y,is_predict = False,action_threhold = 0.01):
    x = x.as_matrix()
    y = y.as_matrix()
 
    pred,_ = model.predict(x)
    idx = pred.argmax(1)
    if not is_predict:
        for i in range(len(idx)):
            prob = np.random.rand()
            if prob < action_threhold:
                idx[i] = np.random.randint(2)
    pred = pred[idx]
    return np.sum(pred)


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

def clip(y):
    y = y/100.0
    #y = y/100.0
    #y[y<=REWARD_THREHOLD] = 0
    #y[y>REWARD_THREHOLD] = 1
    #y = y.clip(lower = -5.0,upper = 5.0)
    #y = y.clip(upper = 5.0)
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

if __name__=="__main__":
    main()
