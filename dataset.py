#-*- coding:utf-8 -*-

import sqlite3
import feature

def load_races(db_path,features,typ):
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect(db_path)

    f_orm = feature.Feature(db_con)
    target_columns = features
    for x,y in f_orm.fetch_race(target_columns,typ):
        dataset_x.append(x)
        dataset_y.append(y)
    return dataset_x,dataset_y

def load_horses(db_path,features,typ):
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect(db_path)

    f_orm = feature.Feature(db_con)
    target_columns = features
    for x,y in f_orm.fetch_horse(target_columns,typ):
        dataset_x.append(x)
        dataset_y.append(y)
    return dataset_x,dataset_y

def races_to_horses(x,y):
    #x = x.values.to_list()
    #y = y.values.to_list()
    result_x = []
    result_y = []
    for rx,ry in zip(x,y):
        for hx,hy in zip(rx,ry):
            result_x.append(hx)
            result_y.append(hy)
    return (result_x,result_y)

def pad_race(x,y,n = 18,columns_dict = {}):
    null_x = []
    null_y = []
    for rx,ry in zip(x,y):
        hnumber = len(rx)
        for i in range(N):
            pass

def get_dummies(x,y):
    pass

def fillna_mean(df):
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mean())
    return df
    #return df.dropna(axis=1)

def fillna_zero(df):
    return df.fillna(0)
