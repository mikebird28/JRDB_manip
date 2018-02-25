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
import pickle,argparse,random
import util, data_processor


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
    config = util.get_config("config/xgbc_config2.json")
    db_path = "db/output_v18.db"

    if use_cache:
        print("[*] load dataset from cache")
        datasets = data_processor.load_from_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(db_path,config)
        datasets.save(CACHE_PATH)
    print("")
    print("[*] training step")
    dnn(config.features,datasets)

def generate_dataset(db_path,config):
    print("[*] preprocessing step")
    print(">> loading dataset")

    x_columns = config.features
    y_columns = [
        "is_win","win_payoff",
        "is_place","place_payoff",
    ]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 11"

    dp = data_processor.load_from_database(db_path,x_columns,y_columns,odds_columns,where = where)
    dp.dummy()
    dp.fillna_mean()
    dp.normalize()
    dp.to_race_panel()
    return dp

def dnn(features,datasets):
    pred_model = load_model("./models/mlp_model3.h5")
    train_x = datasets.get(data_processor.KEY_TRAIN_X).as_matrix()
    train_pred = pred_model.predict(train_x)

    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"is_win"]
    train_payoff = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"win_payoff"]

    test_x  = datasets.get(data_processor.KEY_TEST_X).as_matrix()
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,"is_place"]
    test_payoff = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,"place_payoff"]

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

