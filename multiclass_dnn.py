# -*- coding:utf-8 -*-

from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
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
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    save_value(mean,MEAN_PATH)
    save_value(std,STD_PATH)
    train_x = fillna_mean(train_x,mean)
    test_x = fillna_mean(test_x,mean)

    print(">> normalizing datasets")
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

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

    train_x1 = train_x.ix[:20000,:,features]
    train_x2 = train_x.ix[20000:,:,features]
    train_add_x = train_add_x.loc[:,:,additional_features]
    train_y = train_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]
    test_x = test_x.loc[:,:,features]
    test_add_x = test_add_x.loc[:,:,additional_features]
    test_y = test_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]
    del train_x

    datasets = {
        "train_x1" : train_x1,
        "train_x2" : train_x2,
        "train_add_x" : train_add_x,
        "train_y" : train_y,
        "test_x"  : test_x,
        "test_add_x" : test_add_x,
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
    #train_add_x = datasets["train_add_x"].as_matrix()
    train_y = datasets["train_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_x = datasets["test_x"].as_matrix()
    #test_add_x = datasets["test_add_x"].loc[:,:,"linfo_place_odds"].as_matrix().T
    test_add_x = datasets["test_add_x"].loc[:,:,"linfo_win_odds"].as_matrix().T
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
        win_eval  = evaluate.top_n_k_keras(model,test_x,test_y,test_r_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k_keras(model,test_x,test_r_place,test_r_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

        index_simulator(model,test_x,test_r_win,test_add_x)
    print("")
    print("Accuracy: {0}".format(accuracy))


def create_model(activation = "relu",dropout = 0.4,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    feature_size = 287
    l2_coef = 0.0
    bn_axis = 1
    momentum = 0
    inputs = Input(shape = (18,feature_size,))
    x = inputs
    x = GaussianNoise(0.01)(x)
    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)

    depth = 32
    x = Conv2D(depth,(1,feature_size),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Dropout(dropout)(x)

    """
    race_depth = 8
    tmp = Permute((2,3,1),input_shape = (1,18,depth))(x)
    tmp = Conv2D(race_depth,(1,depth),padding = "valid",kernel_regularizer = l2(l2_coef))(tmp)
    tmp = Activation(activation)(tmp)
    tmp = Conv2D(race_depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(tmp)
    tmp = Dropout(0.8)(tmp)
    tmp = Lambda(lambda x : K.tile(x,(1,18,1,1)))(tmp)
    x = Concatenate(axis = 3)([x,tmp])
    depth = depth + race_depth
    """

    for i in range(1):
        res = x
        x = Conv2D(depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)

        x = Conv2D(depth,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Dropout(dropout)(x)
        x = Add()([x,res])

    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Conv2D(12,(1,1),padding = "valid",kernel_regularizer = l2(l2_coef))(x)
    x = Flatten()(x)
    x = Dense(units = 18)(x)
    #x = GlobalMaxPooling2D(data_format = "channels_first")(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
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

def index_simulator(model,x,y,odds):
    index = mk_index(model,x,odds)
    minimum = 0.0
    maximum = np.max(index)
    delta = (maximum - minimum)/20
    for i in drange(minimum,maximum,delta):
        descrete = np.zeros_like(index)
        descrete[index>i] = 1
        buy_num = np.sum(descrete)
        hit = np.sum(np.clip(descrete * y,0,1))/buy_num
        rewards = np.sum(descrete * y)
        rewards_per = rewards/buy_num
        print("[*] index {0} : buy {1}, hit {2}, rewards_per {3}".format(i,buy_num,hit,rewards_per)) 

def mk_index(model,x,odds):
    offset = 0.5
    #offset = np.e
    clipped = np.clip(odds,1e-8,20) + offset
    #log_odds = np.log(clipped)
    log_odds = np.log(clipped)/np.log(3)
    pred = model.predict(x)
    ret = log_odds * pred
    return ret

def drange(begin, end, step):
    n = begin
    while n+step < end:
        yield n
        n += step

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
