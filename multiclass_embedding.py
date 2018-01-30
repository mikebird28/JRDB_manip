# -*- coding:utf-8 -*-

from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise,SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda
from keras.layers.embeddings import Embedding
import keras.backend as K
import keras.optimizers
import numpy as np
import pandas as pd
import sqlite3, pickle,argparse
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
np.set_printoptions(threshold=np.nan)
 
def main(use_cache = False):
    #predict_type = "place_payoff"
    predict_type = PREDICT_TYPE
    config = util.get_config("config/xgbc_config2.json")
    db_path = "db/output_v17.db"

    db_con = sqlite3.connect(db_path)
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
    dnn(config.features,db_con,datasets)

def generate_dataset(predict_type,db_con,config):
    print("[*] preprocessing step")
    print(">> loading dataset")
    main_features = config.features
    remove_columns = ["info_prize","rinfo_month",]
    #main_features = main_features + ["info_race_id"]
    where = "info_year > 08 and info_year < 90 and rinfo_discipline != 3"
    x,y = dataset2.load_dataset(db_con,main_features,["is_win","win_payoff","is_place","place_payoff"],where = where)

    #additional_x = x.loc[:,additional_features]
    #x = x.loc[:,main_features]

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    #x = dataset2.get_dummies(x,col_dic)

    main_features = sorted(x.columns.values.tolist())

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y,test_nums = 1000)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)
    del x,y
    
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

    print(">> generating target variable")
    train_y["is_win"] = train_y["win_payoff"].clip(lower = 0,upper = 1)
    train_y["is_place"] = train_y["place_payoff"].clip(lower = 0,upper = 1)
    test_y["is_win"] = test_y["win_payoff"].clip(lower = 0,upper = 1)
    test_y["is_place"] = test_y["place_payoff"].clip(lower = 0,upper = 1)

    print(">> converting train dataset to race panel")
    features = sorted(train_x.columns.drop("info_race_id").values.tolist())
    train_x,train_y = dataset2.pad_race(train_x,train_y)
    test_x,test_y = dataset2.pad_race(test_x,test_y)
    train_x = dataset2.downcast(train_x)
    train_y = dataset2.downcast(train_y)
    train_x,train_y = dataset2.to_race_panel(train_x,train_y)
    test_x,test_y = dataset2.to_race_panel(test_x,test_y)

    train_x1 = train_x.ix[:20000,:,features]
    train_x2 = train_x.ix[20000:,:,features]
    train_y = train_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]
    test_x = test_x.loc[:,:,features]
    test_y = test_y.loc[:,:,["place_payoff","win_payoff","is_win","is_place"]]
    del train_x

    datasets = {
        "train_x1" : train_x1,
        "train_x2" : train_x2,
        "train_y" : train_y,
        "test_x"  : test_x,
        "test_y"  : test_y,
    }
    dataset2.save_cache(datasets,CACHE_PATH)
    return datasets

def dnn(features,db_con,datasets):
    print("[*] training step")
    target_y = "is_win"
    #target_y = "is_place"
    train_x = pd.concat([datasets["train_x1"],datasets["train_x2"]],axis = 0)
    test_x = datasets["test_x"]
    train_y = datasets["train_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    #test_add_x = datasets["test_add_x"].loc[:,:,"linfo_place_odds"].as_matrix().T
    test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win = datasets["test_y"].loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T
    test_r_place = datasets["test_y"].loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    sorted_columns = sorted(train_x.axes[2].values.tolist())
    categorical_dic = dataset2.nominal_columns(db_con)

    print(train_x.shape)
    train_x = [train_x.loc[:,:,col].T.as_matrix() for col in sorted_columns]
    test_x = [test_x.loc[:,:,col].T.as_matrix() for col in sorted_columns]
    """
    where = "info_year > 08 and info_year < 90 and rinfo_discipline != 3 and rinfo_race_requirements != 8 and rinfo_race_requirements != 9"
    for idx in test_rx.axes[0]:
        tmp = test_rx.ix[idx,:,:]
        col_ls = [tmp.loc[:,col].as_matrix() for col in sorted_columns]
        new_test_rx.append(col_ls)
    test_rx = new_test_rx
    """

    model =  create_model(categorical_dic,sorted_columns)
    for i in range(1000):
        print("Epoch : {0}".format(i))
        model.fit(train_x,train_y,epochs = 1,batch_size = 2000)
        loss,accuracy = model.evaluate(test_x,test_y,verbose = 0)
        print(loss)
        print(accuracy)
        """
        win_eval  = evaluate.top_n_k_keras(model,test_x,test_y,test_r_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k_keras(model,test_x,test_r_place,test_r_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
        save_model(model,MODEL_PATH)
        train_x,train_y = shuffle_dataset(train_x,train_y)

        #index_simulator(model,test_x,test_r_win,test_add_x)
        """
    print("")
    print("Accuracy: {0}".format(accuracy))


def create_model(categorical_dic,sorted_columns,activation = "relu",dropout = 0.3,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    feature_size = 368
    l2_coef = 0.0
    bn_axis = -1
    momentum = 0
    """
    inputs = Input(shape = (18,feature_size,))
    x = inputs
    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)
    x = GaussianNoise(0.01)(x)
    """
    inputs = []
    flatten_layers = []
    for i,col in enumerate(sorted_columns):

         if col in categorical_dic.keys():
             x = Input(shape = (18,))
             inputs.append(x)
             inp_dim = categorical_dic[col] + 1
             out_dim = max(inp_dim//10,1)
             x = Lambda(lambda a : K.clip(a,0,inp_dim))(x)
             x = Embedding(inp_dim+1,out_dim,input_length = 18)(x)
             x = SpatialDropout1D(0.5)(x)
             #x = Flatten()(x)
             #x = Reshape([18,out_dim,1])(x)
         else:
             x = Input(shape = (18,))
             inputs.append(x)
             x = Reshape([18,1])(x)
         flatten_layers.append(x)
    x = Concatenate(axis = 2)(flatten_layers)
    print(x.shape[1])
    x = Reshape([x.shape[1].value,x.shape[2].value,1])(x)
    #x = Reshape([x.shape[0],x.shape[1],1])(x)
    depth = 60

    x = Conv2D(depth,(1,x.shape[2].value),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Dropout(dropout)(x)

    for i in range(2):
        res = x
        x = ZeroPadding2D(padding = ((0,1),(0,0)))(x)
        x = Conv2D(depth,(2,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Activation(activation)(x)

        x = Conv2D(depth,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        #x = Activation(activation)(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Dropout(dropout)(x)
        x = Add()([x,res])
        x = Activation(activation)(x)

    x = BatchNormalization(axis = 1,momentum = momentum)(x)
    x = Conv2D(1,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = Flatten()(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=1e-3)
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

def shuffle_dataset(np_x,np_y):
    idx = np.random.permutation(18)
    np_x = np_x[:,idx,:]
    np_y = np_y[:,idx]
    return (np_x,np_y)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
