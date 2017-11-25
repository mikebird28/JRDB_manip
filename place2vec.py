#-*- coding:utf-8 -*-

from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1,l2
import keras.backend as K
import keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import util
import sqlite3
import dataset2
import itertools
import argparse
import evaluate

CACHE_PATH  = "./cache/place2vec"
MODEL_PATH = "./models/place2vec.h5"

def main(use_cache = False):
    #predict_type = "is_win"
    predict_type = "is_place"
    config = util.get_config("config/config.json")
    db_path = "db/output_v9.db"
    db_con = sqlite3.connect(db_path)

    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(datasets)

def get_vector(x1,x2,x3,prefix = "p2v"):
    x1 = x1.to_frame()
    x2 = x2.to_frame()
    x3 = x3.to_frame()
    x3 = dist_to_norm(x3)

    x1.reset_index(drop = True,inplace = True)
    x2.reset_index(drop = True,inplace = True)
    x3.reset_index(drop = True,inplace = True)
    x = pd.concat([x1,x2],axis = 1)
    x.columns = ["x1","x2","x3"]
    x["x1"] = x["x1"] - 1
    x["x2"] = x["x2"] - 1
    x["x3"] = x["x3"] - 1
    x["target"] = x["x3"] * 33 +x["x1"] * 3 + x["x2"]
 
    x = dataset2.get_dummies(x["target"],col_dic = {"target":165}).as_matrix()
    model = load_model(MODEL_PATH)
    vectors = pd.DataFrame(model.predict(x))
    columns = ["{0}_{1}".format(prefix,i) for i in range(len(vectors.columns))]
    vectors.columns = columns
    return vectors

def generate_dataset(predict_type,db_con,config):

    print(">> loading dataset")
    features = ["info_pedigree_id","info_race_course_code","rinfo_discipline","rinfo_distance"]
    target = "target"

    x,y = dataset2.load_dataset(db_con,features,[predict_type])
    x["rinfo_distance"] = dist_to_norm(x["rinfo_distance"])
    x = x[x["info_race_course_code"] != 0] 
    x = x[x["rinfo_discipline"] != 0] 
    x = x[x["rinfo_distance"] != 0] 

    x["info_race_course_code"] = x["info_race_course_code"] - 1
    x["rinfo_discipline"] = x["rinfo_discipline"] - 1
    x["rinfo_distance"] = x["rinfo_distance"] - 1
    x["target"] = x["rinfo_distance"] * 33 + x["info_race_course_code"] * 3 + x["rinfo_discipline"]
    con = pd.concat([x,y],axis = 1)
    con = con[con[predict_type] == 1]

    pairs = []
    for name,horse in con.groupby("info_pedigree_id"):
        course_codes = list(horse[target].unique())
        if len(course_codes) < 2:
            continue
        comb = itertools.permutations(course_codes,2)
        pairs.extend(comb)
    pairs = pd.DataFrame(pairs,columns = ["x","y"])
    pairs = pairs.sample(frac=1.0).reset_index(drop = True)

    x = pairs.loc[:,"x"]
    y = pairs.loc[:,"y"]

    x = dataset2.get_dummies(x,col_dic = {"x":164})
    y = dataset2.get_dummies(y,col_dic = {"y":164})

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 1000)

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
    }
    return datasets


def create_model(input_dim = 165,activation = "relu",dropout = 0.5,hidden_1 = 15):
    nn = Sequential()

    #nn.add(Dense(units=hidden_1,input_dim = input_dim,name = "internal"))
    nn.add(Dense(units=hidden_1,input_dim = input_dim,activity_regularizer = l2(0.0)))
    nn.add(Dense(units=hidden_1,input_dim = input_dim,name = "internal"))
    #nn.add(Activation(activation))
    #nn.add(BatchNormalization(name = "internal"))
    #nn.add(Dropout(dropout))

    nn.add(Dense(units = input_dim))
    nn.add(Activation("softmax"))

    opt = keras.optimizers.Adam(lr=0.01)
    nn.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    return nn

def dnn(datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])

    #model = KerasClassifier(create_model,batch_size = 300,verbose = 1)
    model = create_model()
    internal = Model(inputs = model.input,outputs = model.get_layer("internal").output)
    for i in range(10):
        model.fit(train_x,train_y,epochs = 1,batch_size = 300)
        score = model.evaluate(test_x,test_y,verbose = 0)
        print("test loss : {0}".format(score[0]))
    show_similarity(165,internal)
    save_model(internal,MODEL_PATH)

def dist_to_norm(dist_array):
    def f(x):
        if x < 1400:
            return 1
        elif x<1800:
            return 2
        elif x < 2200:
            return 3
        elif x < 2800:
            return 4
        elif x >= 2800:
            return 5
        else:
            return 0
    return dist_array.apply(f)

def save_model(model,path):
    print("Save model")
    model.save(path)

def show_similarity(inputs_size,model):
    inputs = np.eye(inputs_size)
    outputs = model.predict(inputs)
    evaluate.show_similarity(21,outputs)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
