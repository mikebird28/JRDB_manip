# -*- coding:utf-8 -*-

from keras.regularizers import l1,l2
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda
import keras.optimizers
import numpy as np
import pandas as pd
import pickle,argparse
import data_processor, util, evaluate


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
    config = util.get_config("config/xgbc_config2.json")
    db_path = "db/output_v18.db"
    if use_cache:
        print("[*] load dataset from cache")
        datasets = data_processor.load_from_cache(CACHE_PATH)
    else:
        print("[*] load dataset from database")
        datasets = generate_dataset(db_path,config)
        datasets.save(CACHE_PATH)
    print("[*] training step")
    dnn(config.features,datasets)

def generate_dataset(db_path,config):
    x_columns = config.features
    y_columns = [
       "is_win","win_payoff",
       "is_place","place_payoff",
#       "is_exacta","exacta_payoff",
#       "is_quinella","quinella_payoff",
    ]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 2011"

    dp = data_processor.load_from_database(db_path,x_columns,y_columns,odds_columns,where = where)
    dp.dummy()
    dp.fillna_mean(typ = "race")
    dp.normalize()
    dp.to_race_panel()
    return dp

def dnn(features,datasets):
    target_y = "is_win"
    #target_y = "is_place"
    train_x = datasets.get(data_processor.KEY_TRAIN_X).as_matrix()
    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_x  = datasets.get(data_processor.KEY_TEST_X).as_matrix()
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"is_win"].as_matrix().reshape([18,-1]).T
    test_rp_win  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T
    test_r_place  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"is_place"].as_matrix().reshape([18,-1]).T
    test_rp_place  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    model =  create_model()
    for i in range(1000):
        print("Epoch : {0}".format(i))
        model.fit(train_x,train_y,epochs = 1,batch_size = 2000)
        loss,accuracy = model.evaluate(test_x,test_y,verbose = 0)
        print(loss)
        print(accuracy)
        wrapper = evaluate.KerasMultiWrapper(model)
        win_eval  = evaluate.top_n_k(wrapper,test_x,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k(wrapper,test_x,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
        save_model(model,MODEL_PATH)
        train_x,train_y = shuffle_dataset(train_x,train_y)
    print("")
    print("Accuracy: {0}".format(accuracy))


def create_model(activation = "relu",dropout = 0.3,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
    feature_size = 360
    l2_coef = 0.0
    bn_axis = -1
    momentum = 0
    inputs = Input(shape = (18,feature_size,))
    x = inputs
    x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)
    x = GaussianNoise(0.01)(x)

    depth = 40
    x = Conv2D(depth,(1,feature_size),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Dropout(dropout)(x)

    for i in range(1):
        res = x
        x = ZeroPadding2D(padding = ((0,1),(0,0)))(x)
        x = Conv2D(depth,(2,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Activation(activation)(x)

        x = Conv2D(depth,(1,1),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Dropout(dropout)(x)
        x = Add()([x,res])
        x = Activation(activation)(x)

    x = BatchNormalization(axis = 1,momentum = momentum)(x)
    x = Permute((1,3,2))(x)
    x = Reshape([18,40])(x)
    x = LSTM(units = 10,return_sequences = True)(x)
    x = Reshape([18,10,1])(x)
    x = BatchNormalization(axis = 1,momentum = momentum)(x)
    x = Conv2D(1,(1,10),padding = "valid",kernel_initializer="he_normal",kernel_regularizer = l2(l2_coef))(x)
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
