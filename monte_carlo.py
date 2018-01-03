# -*- coding:utf-8 -*-

from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.models import model_from_config
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda
from keras.layers.pooling import GlobalAveragePooling2D,GlobalMaxPooling2D
import keras.optimizers
import numpy as np
import pandas as pd
import sqlite3, pickle,argparse,decimal
import dataset2, util, evaluate
import place2vec, course2vec,field_fitness
import mutual_preprocess


EVALUATE_INTERVAL = 10
MAX_ITERATION = 50000
BATCH_SIZE = 36
REFRESH_INTERVAL = 50 
PREDICT_TYPE = "win_payoff"
#PREDICT_TYPE = "place_payoff"
MODEL_PATH = "./models/montecarlo.h5"
MEAN_PATH = "./models/montecarlo.pickle"
STD_PATH = "./models/montecarlo.pickle"
CACHE_PATH = "./cache/montecarlo"

pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
np.set_printoptions(threshold=np.nan)
 
def main(use_cache = False):
    #predict_type = "place_payoff"
    predict_type = PREDICT_TYPE
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
    main_features = config.features
    additional_features = ["linfo_win_odds","linfo_place_odds"]
    load_features = main_features + additional_features
    x,p2v,y = mutual_preprocess.load_datasets_with_p2v(db_con,load_features)
    main_features = main_features + ["info_race_id"]

    additional_x = x.loc[:,additional_features]
    x = x.loc[:,main_features]

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

    main_features = sorted(x.columns.values.tolist())
    main_features_dropped = sorted(x.columns.drop("info_race_id").values.tolist())
    additional_features = sorted(additional_x.columns)
    p2v_features = sorted(p2v.columns.values.tolist())

    x = concat(x,p2v)
    x = dataset2.downcast(x)
    del p2v

    print(">> separating dataset")
    con = concat(x,additional_x)
    train_con,test_con,train_y,test_y = dataset2.split_with_race(con,y,test_nums = 1000)
    train_x = train_con.loc[:,x.columns]
    train_add_x = train_con.loc[:,additional_x.columns]
    test_x = test_con.loc[:,x.columns]
    test_add_x = test_con.loc[:,additional_x.columns]
    train_x = dataset2.downcast(train_x)

    test_x = dataset2.downcast(test_x)
    del x,y,con,train_con,test_con
    
    print(">> filling none value of datasets")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-2)
    mean_odds = train_add_x.mean(numeric_only = True)
    #mean_std = train_add_x.std(numeric_only = True).clip(lower = 1e-2)
    save_value(mean,MEAN_PATH)
    save_value(std,STD_PATH)

    train_x = fillna_mean(train_x,mean)
    train_add_x = fillna_mean(train_add_x,mean_odds)
    test_x = fillna_mean(test_x,mean)
    test_add_x = fillna_mean(test_add_x,mean_odds)

    print(">> normalizing datasets")
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)
    train_add_x = (train_add_x/100.0).clip(upper = 50)
    test_add_x = (test_add_x/100.0).clip(upper = 50)

    print(">> adding extra information to datasets")
    train_x,train_y = add_c2v(train_x,train_y,main_features_dropped,nom_col)
    test_x,test_y = add_c2v(test_x,test_y,main_features_dropped,nom_col)
    train_x,train_y = add_ff(train_x,train_y,main_features_dropped,nom_col)
    test_x,test_y = add_ff(test_x,test_y,main_features_dropped,nom_col)

    print(">> generating target variable")
    train_y["is_win"] = train_y["win_payoff"].clip(lower = 0,upper = 1)
    train_y["is_place"] = train_y["place_payoff"].clip(lower = 0,upper = 1)
    test_y["is_win"] = test_y["win_payoff"].clip(lower = 0,upper = 1)
    test_y["is_place"] = test_y["place_payoff"].clip(lower = 0,upper = 1)

    print(">> converting train dataset to race panel")
    features = sorted(train_x.columns.drop("info_race_id").values.tolist())
    train_x,train_y,train_add_x = dataset2.pad_race(train_x,train_y,train_add_x)
    test_x,test_y,test_add_x = dataset2.pad_race(test_x,test_y,test_add_x)
    train_x = dataset2.downcast(train_x)
    train_y = dataset2.downcast(train_y)
    train_x,train_y,train_add_x = dataset2.to_race_panel(train_x,train_y,train_add_x)
    test_x,test_y,test_add_x = dataset2.to_race_panel(test_x,test_y,test_add_x)

    train_x = train_x.loc[:,:,features]
    train_add_x = train_add_x.loc[:,:,additional_features]
    train_y = train_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]
    test_x = test_x.loc[:,:,features]
    test_add_x = test_add_x.loc[:,:,additional_features]
    test_y = test_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]

    print(">> prediting with dnn ")
    train_race_idx = train_x.axes[0]
    test_race_idx = test_x.axes[0]
    train_x = train_x.as_matrix()
    test_x = test_x.as_matrix()
    pred_model = load_model("./models/mlp_model3.h5")

    train_x = pred_model.predict(train_x)
    train_x = pd.Panel({"pred":pd.DataFrame(train_x,index = train_race_idx)})
    train_x = train_x.swapaxes(0,1,copy = False)
    train_x = train_x.swapaxes(1,2,copy = False)
    train_x = pd.concat([train_x,train_add_x],axis = 2)

    test_x = pred_model.predict(test_x)
    test_x = pd.Panel({"pred":pd.DataFrame(test_x,index = test_race_idx)})
    test_x = test_x.swapaxes(0,1,copy = False)
    test_x = test_x.swapaxes(1,2,copy = False)
    test_x = pd.concat([test_x,test_add_x],axis = 2)
    print(train_x.shape)
    print(test_x.shape)

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
    train_x = datasets["train_x"]
    train_r_win = datasets["train_y"].loc[:,:,["win_payoff"]] -100
    #train_r_win = train_r_win.apply(lambda x: -100 if x == 0 else x)
    #train_y = datasets["train_y"].loc[:,:,target_y]
    test_x = datasets["test_x"].as_matrix()
    #test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win = datasets["test_y"].loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T-100
    #test_r_place = datasets["test_y"].loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    model =  create_model()
    old_model = clone_model(model)
    gene = dataset_generator(train_x,train_r_win,batch_size = 100)
    
    #constants
    max_iterate = 1000
    log_interval = 10
    model_swap_interval = 5

    #variables
    skip_count = 0 
    hit_count = 0

    for i in range(50000):
        x,y = next(gene)
        x = x.as_matrix()
        y = y.as_matrix()
        y = y.reshape([-1,18],order='F')

        sample_count = 0
        count = 0
        pred = old_model.predict(x)
        while True:
            count += 1
            noise = (float(count)/max_iterate)/3
            n_pred = np.clip(pred,0.1,0.9)
            #n_pred = np.clip(pred,noise,1-noise)
            act = np.random.binomial(1,n_pred)
            eval_v = eval_action(act,y,100,)
            if eval_v == 1:
                sample_count += 1
                model.fit(x,act,epochs = 1,verbose = 0)
                hit_count += 1
            if sample_count == 1:
                break
            if count >= max_iterate:
                skip_count += 1
                break
        test_pred = model.predict(test_x)
        test_pred = np.round(test_pred)
        total_reward = np.sum(test_pred*test_r_win)
        buy = np.sum(test_pred)
        reward_per = float(total_reward/buy)
        if i%log_interval == 0:
            print("Epoch {0}, Buy : {1}, RewardPer : {2}".format(i,buy,reward_per))
            print("Skip Count: {0}".format(skip_count))
            skip_count = 0
        if hit_count == model_swap_interval:
            old_model = clone_model(model)
            hit_count = 0


def eval_action(action,payoff,batch_size):
    buy = np.sum(action)
    reward_matrix = action*payoff
    hit_matrix = np.clip(reward_matrix,0,1)
    total_reward = np.sum(reward_matrix)
    total_hit = np.sum(hit_matrix)
    reward_per = total_reward/buy if buy != 0 else 0
    hit_per = total_hit/batch_size
    if reward_per > 10.0:
        return 1
    else:
        return 0

def create_model(activation = "relu",dropout = 0.4,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    inputs = Input(shape = (18,3))
    x = inputs
    x = Flatten()(x)
    x = Dense(units = 36)(x)
    #outputs = Activation(activation)(x)
    #x = Activation(activation)(x)
    x = Dropout(0.8)(x)
    x = Dense(units = 18)(x)
    outputs = Activation("sigmoid")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss = "mse",optimizer=opt,metrics=["accuracy"])
    #model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model

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

def to_descrete(array):
    res = np.zeros_like(array)
    res[array.argmax(1)] = 1
    return res

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

def add_c2v(x,y,features,nom_col):
    c2v_x = x.loc[:,features]
    c2v_df = course2vec.get_vector(c2v_x,nom_col)
    x = concat(x,c2v_df)
    y.reset_index(inplace = True,drop = True)
    del c2v_x
    del c2v_df
    return (x,y)

def add_ff(x,y,features,nom_col):
    ff_x = x.loc[:,features]
    ff_df = field_fitness.get_vector(ff_x,nom_col)
    x = concat(x,ff_df)
    y.reset_index(drop = True,inplace = True)
    del ff_x
    del ff_df
    return (x,y)

def fillna_mean(df,mean = None):
    if type(mean) == type(None):
        mean = df.mean(numeric_only = True)
        df = df.fillna(mean)
    else:
        df = df.fillna(mean)
    return df

def drange(begin, end, step):
    n = begin
    while n+step < end:
        yield n
        n += step

def dataset_generator(x,y,batch_size = 100):
    x_col = x.axes[2].tolist()
    y_col = y.axes[2].tolist()
 
    y.axes[0] = x.axes[0]
    y.axes[1] = x.axes[1]
    con = pd.concat([x,y],axis = 2)
    while True:
        sample = con.sample(n = batch_size,axis = 0)
        x = sample.loc[:,:,x_col]
        y = sample.loc[:,:,y_col]
        yield (x,y)

def clone_model(model,custom_objects = {}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
