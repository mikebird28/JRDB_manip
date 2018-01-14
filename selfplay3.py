# -*- coding:utf-8 -*-

#
#Let us pray the success of experiments
#

import keras.backend as K
from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise
from keras.models import model_from_config
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda
from keras.layers.pooling import GlobalAveragePooling2D,GlobalMaxPooling2D
import keras.optimizers
import numpy as np
import pandas as pd
import sqlite3, pickle,argparse,sys,random
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
    target_y = "place_payoff"
    #target_y = "win_payoff"
    train_x = datasets["train_x"]
    #print(train_x.columns)
    train_y = datasets["train_y"].loc[:,:,[target_y]] -100

    print(train_y.shape)
    test_x = datasets["test_x"].as_matrix()
    #test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T-100
    test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().T-100
    print(test_y.shape)
    print(test_y[0,:])
    #test_y = test_y.reshape([-1,18])
    #print(test_y[0,:])
   
    max_iterate = 300000
    log_interval = 2
    model_swap_interval = 10
    model =  create_model()
    old_model = clone_model(model)
    sampler = ActionSampler(train_x,train_y,max_sampling = 500)

    for i in range(max_iterate):
        target_size = 10
        batch_size = 500
        x,y,act = sampler.get_sample(old_model,target_size,batch_size)
        model.fit(x,act,epochs = 1,batch_size = batch_size,verbose = 0)
        if i%log_interval == 0:
            log(i,model,x,y)
            log(i,model,test_x,test_y)
        if i%model_swap_interval == 0:
            print("swap")
            old_model = clone_model(model)

class ActionSampler():

    def __init__(self,x,y,max_sampling = 1000):
        self.max_sampling = max_sampling
        self.x = x
        self.y = y
        self.con = pd.concat([self.x,self.y],axis = 2)

        self.x_col = x.axes[2].tolist()
        self.y_col = y.axes[2].tolist()
        self.y.axes[0] = self.x.axes[0]
        self.y.axes[1] = self.x.axes[1]
 
    def get_sample(self,model,target_size,batch_size):
        timeout = 10000
        sample_list_x = []
        sample_list_y = []
        sample_list_act = []
        for i in range(timeout):
            x,y = self.__random_sample(target_size)
            pred = model.predict(x)
            pred_act = np.eye(2)[pred_to_act(pred)]
            pred_value = eval_action(pred_act,y,target_size)

            best_value = pred_value
            best_act = pred_act
            should_push = False
            hit_count = 1.
            for j in range(self.max_sampling):
                random_act = self.__generate_action(pred)
                eval_value = eval_action(random_act,y,target_size)
                if eval_value > pred_value:
                #if eval_value > best_value:
                    #best_value = eval_value
                    #best_act = random_act
                    best_act += random_act
                    hit_count += 1
                    should_push = True
            if should_push:
                best_act = best_act/hit_count
                sample_list_x.append(x)
                sample_list_y.append(y)
                sample_list_act.append(best_act)
            now_len = len(sample_list_x) * target_size
            if now_len >= batch_size:
                break
        ret_x = np.concatenate(sample_list_x,axis = 0)
        ret_y = np.concatenate(sample_list_y,axis = 0)
        ret_act = np.concatenate(sample_list_act,axis = 0)
        #ret_act = np.eye(2)[ret_act]
        return (ret_x,ret_y,ret_act)

    def __generate_action(self,pred):
        dice = random.random()
        if dice < 0.7:
            random_act = random_choice(pred)
        else:
            random_act = random_choice_equally(pred)
        random_act = np.eye(2)[random_act]
        return random_act

    def __random_sample(self,batch_size):
        sample = self.con.sample(n = batch_size,axis = 0)
        x = sample.loc[:,:,self.x_col]
        y = sample.loc[:,:,self.y_col]

        x = x.as_matrix()
        y = y.as_matrix()
        y = y.reshape([-1,18],order='F')
        return (x,y)


def eval_action(action,payoff,batch_size):
    action = np.argmax(action,2)
    #buy = np.sum(action)
    reward_matrix = action*payoff
    #hit_matrix = np.clip(reward_matrix,0,1)
    total_reward = np.sum(reward_matrix)
    #total_hit = np.sum(hit_matrix)
    #reward_per = total_reward/buy if buy != 0 else 0
    #hit_per = total_hit/batch_size
    #return max(reward_per,10.0)
    #return reward_per
    return max(total_reward,0.0)

def log(epoch,model,x,y):
    pred = model.predict(x)
    print(pred[0,:])
    #pred_act = np.round(pred)
    #pred_act = to_descrete(pred)
    pred_act = pred_to_act(pred)
    #print(pred)
    reward_matrix = pred_act*y
    hit_matrix = np.clip(reward_matrix,0,1)
    total_hit = np.sum(hit_matrix)
    total_reward = np.sum(reward_matrix)
    buy = np.sum(pred_act)
    reward_per = float(total_reward/buy)
    print("Epoch {0}, Hit : {1}/{2}, Reward Per : {3}, Total Reward : {4}".format(epoch,total_hit,buy,reward_per,total_reward))


def create_model(activation = "relu",dropout = 0.4,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    l2_coef = 0
    l2_coef = 1e-6
    feature_size = 3
    inputs = Input(shape = (18,3))
    bn_axis = -1
    depth = 12
    x = inputs
    x = GaussianNoise(0.01)(x)

    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)
    x = Conv2D(depth,(1,3),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = BatchNormalization(momentum = 0)(x)

    for i in range(3):
        tmp = x
        x = Conv2D(depth,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization(axis = bn_axis,momentum = 0)(x)
        x = Activation(activation)(x)
        x = Conv2D(depth,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization(axis = bn_axis,momentum = 0)(x)
        #x = Dropout(0.7)(x)
        x = Add()([tmp,x])
        x = Activation(activation)(x)

    x = BatchNormalization(axis = 1,momentum = 0)(x)
    x = Conv2D(2,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    #x = Dense(units = 36)(x)
    #x = Reshape([18,2],input_shape = (18*2,))(x)

    x = Permute((1,3,2))(x)
    x = Reshape([18,2],input_shape = (18,2,1))(x)
    #x = Activation(activation)(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    #model.compile(loss = log_loss,optimizer=opt,metrics=["accuracy"])
    #model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
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
    res[range(len(array)),array.argmax(1)] = 1
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


def pred_to_act(pred):
    act = np.argmax(pred,2)
    return act

def binomial_action(pred):
    act = np.random.binomial(1,pred)
    return act

def random_inverts(p,matrix):
    matrix = matrix.astype(bool)
    random_mask = np.random.binomial(1,p,size = matrix.shape).astype(bool)
    random_value = np.random.randint(2,size = matrix.shape)
    ret = np.where(random_mask == True,random_value,matrix).astype(np.int32)
    return ret

def random_choice(matrix):
    matrix = matrix - 0.001
    def f(p):
        return np.random.multinomial(1,p)
    ret = np.apply_along_axis(f,2,matrix)
    ret = np.argmax(ret,2)
    return ret

def random_choice_equally(matrix):
    a = np.ones(matrix.shape)/2.
    return random_choice(a)

def clone_model(model,custom_objects = {}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def log_loss(y_true,y_pred):
    y_pred = K.clip(y_pred,1e-8,1.0)
    return -y_true * K.log(y_pred)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)

