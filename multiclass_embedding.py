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
import pickle,argparse
import util, data_processor, evaluate


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
    print(">> loading dataset")
    x_columns = config.features
    y_columns = [
        "is_win","win_payoff",
        "is_place","place_payoff",
#        "is_exacta","exacta_payoff",
#        "is_quinella","quinella_payoff",
        ]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 2011"
    dp = data_processor.load_from_database(db_path,x_columns,y_columns,odds_columns,where = where)
    dp.fillna_mean(typ = "race")
    dp.normalize()
    dp.to_race_panel()
    return dp

def dnn(features,datasets):

    target_y = "is_win"
    #target_y = "is_place"
    train_x = datasets.get(data_processor.KEY_TRAIN_X)
    train_y = datasets.get(data_processor.KEY_TRAIN_Y).loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_x  = datasets.get(data_processor.KEY_TEST_X)
    test_y  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,target_y].as_matrix().reshape([18,-1]).T
    test_r_win  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"is_win"].as_matrix().reshape([18,-1]).T
    test_rp_win  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"win_payoff"].as_matrix().reshape([18,-1]).T
    test_r_place  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"is_place"].as_matrix().reshape([18,-1]).T
    test_rp_place  = datasets.get(data_processor.KEY_TEST_Y).loc[:,:,"place_payoff"].as_matrix().reshape([18,-1]).T

    categorical_dic = datasets.categorical_columns
    sorted_columns = sorted(train_x.axes[2].values.tolist())
    train_x = [train_x.loc[:,:,col].T.as_matrix() for col in sorted_columns]
    test_x_s = [test_x.loc[:,:,col].T.as_matrix() for col in sorted_columns]

    model =  create_model(categorical_dic,sorted_columns)
    for i in range(1000):
        print("Epoch : {0}".format(i))
        model.fit(train_x,train_y,epochs = 1,batch_size = 2000)
        loss,accuracy = model.evaluate(test_x_s,test_y,verbose = 0)
        print(loss)
        print(accuracy)
        wrapper = evaluate.KerasMultiEmbbedWrapper(model)
        """
        win_eval  = evaluate.top_n_k(wrapper,test_x,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k(wrapper,test_x,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
        """
        save_model(model,MODEL_PATH)

    print("")
    print("Accuracy: {0}".format(accuracy))


def create_model(categorical_dic,sorted_columns,activation = "relu",dropout = 0.3,hidden_1 = 80,hidden_2 = 80,hidden_3 = 80):
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
