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

CACHE_PATH  = "./cache/place2vec"
MODEL_PATH = "./models/place2vec.h5"

def main(use_cache = False):
    predict_type = "is_place"
    config = util.get_config("config/config.json")
    db_path = "db/output_v7.db"
    db_con = sqlite3.connect(db_path)
 
    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(datasets)

def get_vector(x):
    x = pd.get_dummies(x["info_race_course_code"]).as_matrix()
    model = load_model(MODEL_PATH)
    vectors = pd.DataFrame(model.predict(x))
    columns = ["p2v_{0}".format(i) for i in range(len(vectors.columns))]
    vectors.columns = columns
    return vectors

def generate_dataset(predict_type,db_con,config):

    print(">> loading dataset")
    features = ["info_pedigree_id","info_race_course_code"]
    x,y = dataset2.load_dataset(db_con,features,[predict_type])
    con = pd.concat([x,y],axis = 1)
    con = con[con[predict_type] == 1]

    pairs = []
    for name,horse in con.groupby("info_pedigree_id"):
        course_codes = list(horse["info_race_course_code"].unique())
        if len(course_codes) < 2:
            continue
        comb = itertools.permutations(course_codes,2)
        pairs.extend(comb)
    pairs = pd.DataFrame(pairs,columns = ["x","y"])
    pairs = pairs.sample(frac=1.0).reset_index(drop = True)

    x = pairs.loc[:,"x"].to_frame()
    y = pairs.loc[:,"y"].to_frame()

    x = pd.get_dummies(x["x"])
    y = pd.get_dummies(y["y"])

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 1000)

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
    }
    return datasets


def create_model(activation = "relu",dropout = 0.5,hidden_1 = 15):
    nn = Sequential()

    nn.add(Dense(units=hidden_1,input_dim = 10,name = "internal"))
    #nn.add(Activation(activation))
    nn.add(BatchNormalization())
    #nn.add(Dropout(dropout))

    """
    nn.add(Activation(activation))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout))
    """

    nn.add(Dense(units = 10))
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
        print(i)
        model.fit(train_x,train_y,epochs = 1,batch_size = 300)
        score = model.evaluate(test_x,test_y,verbose = 0)
        print("test loss : {0}".format(score[0]))
        p = internal.predict(test_x,verbose = 0)
        print(p)
    save_model(internal,MODEL_PATH)

def save_model(model,path):
    print("Save model")
    model.save(path)

if __name__=="__main__":
    main()



