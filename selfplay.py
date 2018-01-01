# -*- coding:utf-8 -*-

#
#Let us pray the success of experiments
#

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
    target_y = "is_win"
    train_x = datasets["train_x"]
    train_r_win = datasets["train_y"].loc[:,:,["win_payoff"]] -100
    #train_r_win = train_r_win.apply(lambda x: -100 if x == 0 else x)
    #train_y = datasets["train_y"].loc[:,:,target_y]
    test_x = datasets["test_x"].as_matrix()
    #test_y = datasets["test_y"].loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win = datasets["test_y"].loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T-100
    #test_r_place = datasets["test_y"].loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

   
    #constants
    max_iterate = 300000
    max_sampling = 3000
    log_interval = 10
    model_swap_interval = 20
    batch_size = 200
    hist = None

    #variables
    skip_count = 0 
    swap_count = 0
    hit_count = 0

    model =  create_model()
    old_model = clone_model(model)
    gene = dataset_generator(train_x,train_r_win,batch_size = batch_size)
    sampler = ActionSampler(train_x,train_r_win)
 
    for i in range(max_iterate):
        sampler.get_sample(mode,target_size,batch_size)

        """
        x,y = next(gene)
        x = x.as_matrix()
        y = y.as_matrix()
        y = y.reshape([-1,18],order='F')

        pred = old_model.predict(x)
        pred_act = np.round(pred)
        pred_v = eval_action(pred_act,y,batch_size)
        best_value = pred_v
        best_act = pred_act
        best_count = 1
        should_train = False
        for j in range(max_sampling):
            dice = random.random()
            if dice < 0.95:
                #random_act = np.random.randint(2,size = pred.shape)
                #random_act = random_inverts(0.03,pred_act)
                max_value = 10000.0
                prob = max((max_value-i)/max_value, 5e-2)
                random_act = random_inverts(prob,pred_act)
            else:
                noise = (float(j)/max_iterate)/3
                n_pred = np.clip(pred,noise,0.99-noise)
                random_act = np.random.binomial(1,n_pred)
            eval_v = eval_action(random_act,y,batch_size)
            if eval_v > best_value:
                best_count += 1
                best_value = eval_v
                best_act = random_act
                #best_act += random_act
                should_train = True
        if should_train:
            #best_act = best_act/best_count
            #print(best_act[-1,:])
            #if i < 200:
            #    best_act = np.clip(y,0,1)
            hist = model.fit(x,best_act,epochs = 1,verbose = 0,batch_size = batch_size)
            loss = hist.history["loss"]
            swap_count += 1
        else:
            skip_count += 1
        """
         
        if i%log_interval == 0:
            test_pred = model.predict(test_x)
            test_pred = np.round(test_pred)
            reward_matrix = test_pred*test_r_win
            hit_matrix = np.clip(reward_matrix,0,1)
            total_hit = np.sum(hit_matrix)
            total_reward = np.sum(reward_matrix)
            buy = np.sum(test_pred)
            reward_per = float(total_reward/buy)
            print("Epoch {0}, Hit : {1}/{2}, Reward Per : {3}, Total Reward : {4}".format(i,total_hit,buy,reward_per,total_reward))
            print("skip_count : {0}".format(skip_count))
            skip_count = 0
            #print("loss : {0}".format(loss))

        if swap_count == model_swap_interval:
            print("swap")
            old_model = clone_model(model)
            swap_count = 0

class ActionSampler():

    def __init__(self,x,y,max_sampling = 1000):
        self.max_sampling = max_sampling
        self.x = x
        self.y = y
        self.con = pd.concat([self.x,self.y],axis = 2)

        self.x_col = x.axes[0]
        self.y_col = y.axes[0]
        self.y.axes[0] = self.axes[0]
        self.y.axes[1] = self.axes[1]
 
    def get_sample(self,model,target_size,batch_size):
        if target_size%batch_size != 0:
            raise Exception("target_size should be multiples of batch_size")
        n = target_size/batch_size

        sample_list_x = []
        sample_list_y = []
        for i in range(n):
            x,y = self.__random_sample(batch_size)
            pred = model.predict(x)
            pred_act = np.round(pred)
            pred_value = eval_action(pred_act,y,batch_size)

            best_value = pred_value
            best_act = pred_act
            should_push = False
            for j in range(self.max_sampling):
                random_act = self.__generate_action(pred)
                eval_value = eval_action(random_act,y,batch_size)
                if eval_value > best_value:
                    best_value = eval_value
                    best_act = random_act
                    should_push = True
            if should_push:
                sample_list_x.append(x)
                sample_list_y.append(best_act)
        ret_x = pd.concat(sample_list_x,axis = 0)
        ret_y = pd.concat(sample_list_y,axis = 0)
        print(ret_y.shape)
        return (ret_x,ret_y)

    def __generate_action(pred):
        #pred_act = np.round(pred)
        dice = random.random()
        if dice < 0.95:
            random_act = np.random.randint(2,size = pred.shape)
            #random_act = random_inverts(0.03,pred_act)
            #max_value = 10000.0
            #prob = max((max_value-i)/max_value, 5e-2)
            #random_act = random_inverts(prob,pred_act)
        else:
            #noise = (float(j)/max_iterate)/3
            noise = 0.1
            n_pred = np.clip(pred,noise,0.99-noise)
            random_act = np.random.binomial(1,n_pred)
        return random_act

    def __random_sample(self,batch_size):
        sample = self.con.sample(n = batch_size,axis = 0)
        x = sample.loc[:,:,self.x_col]
        y = sample.loc[:,:,self.y_col]
        yield (x,y)


def eval_action(action,payoff,batch_size):
    #buy = np.sum(action)
    reward_matrix = action*payoff
    #hit_matrix = np.clip(reward_matrix,0,1)
    total_reward = np.sum(reward_matrix)
    #total_hit = np.sum(hit_matrix)
    #reward_per = total_reward/buy
    #hit_per = total_hit/batch_size
    return total_reward
    #return max(total_reward,000)

def create_model(activation = "relu",dropout = 0.4,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    l2_coef = 0.01
    feature_size = 3
    inputs = Input(shape = (18,3))
    x = inputs

    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)
    x = Conv2D(32,(1,3),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = Activation(activation)(x)
    x = BatchNormalization(momentum = 0)(x)
    x = Dropout(0.8)(x)
    x = Conv2D(1,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = Flatten()(x)

    """
    x = Flatten()(inputs)
    #x = GaussianNoise(0.01)(x)

    x = Dense(units = 36)(x)
    x = Activation(activation)(x)
    x = BatchNormalization(momentum = 0)(x)
    x = Dropout(0.6)(x)

    x = Dense(units = 18)(x)
    """
    outputs = Activation("sigmoid")(x)

    model = Model(inputs = inputs,outputs = outputs)
    opt = keras.optimizers.Adam(lr=1e-2)
    #model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    model.compile(loss = "mse",optimizer=opt,metrics=["accuracy"])
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

def random_inverts(p,matrix):
    matrix = matrix.astype(bool)
    random_mask = np.random.binomial(1,p,size = matrix.shape).astype(bool)
    random_value = np.random.randint(2,size = matrix.shape)
    ret = np.where(random_mask == True,random_value,matrix).astype(np.int32)
    return ret

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
    """
    array = np.array([1,1,1,1,0,0,0,0])
    for i in range(10):
        print(random_inverts(0.9,array))
    """
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)

