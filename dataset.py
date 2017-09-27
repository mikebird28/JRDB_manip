#-*- coding:utf-8 -*-

import sqlite3
import feature
import pandas as pd
import numpy as np

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

def races_to_horses(x,y,race_info=None):
    #x = x.values.to_list()
    #y = y.values.to_list()
    result_x = []
    result_y = []
    if race_info is None:
        for rx,ry in zip(x,y):
            for hx,hy in zip(rx,ry):
                result_x.append(hx)
                result_y.append(hy)
    else:
        for rx,ry,ri in zip(x,y,race_info):
            for hx,hy in zip(rx,ry):
                result_x.append(np.append(hx,ri))
                result_y.append(hy)
    return (result_x,result_y)

def pad_race(x,y,n = 18,columns_dict = {}):
    hx,hy = races_to_horses(x,y)
    pd_x = pd.DataFrame(hx)
    mean = pd_x.mean().values.tolist()

    result_x = []
    result_y = []
    count = 0
    for rx,ry in zip(x,y):
        hnumber = len(rx)
        new_x = []
        new_y = []
        for i in range(n):
            try:
                new_x.append(rx[i])
                new_y.append(ry[i])
            except IndexError:
                new_x.append(mean)
                new_y.append(0)
        #new_x = pd.DataFrame(new_x)
        #new_x = fillna_mean(new_x)
        #new_x = fillna_zero(new_x)
        #new_x = new_x.values.tolist()
        result_x.append(new_x)
        result_y.append(new_y)
        if count%100 == 0:
            print(count)
        count += 1
    panel = pd.Panel(result_x)
    panel = panel.fillna(0)
    result_x = panel.values.tolist()
    return (result_x,result_y)


def get_dummies(x,y):
    pass

def fillna_mean(df):
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mean())
    return df
    #return df.dropna(axis=1)

def fillna_zero(df):
    return df.fillna(0)
