#-*- coding:utf-8 -*-

import sqlite3
import feature
import sys
import pandas as pd
import random
import numpy as np

def load_races(db_path,features,typ):
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect(db_path)

    f_orm = feature.Feature(db_con)
    target_columns = features
    for x,y in f_orm.fetch_race(target_columns,typ):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
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
    dataset_x = pd.DataFrame(dataset_x)
    dataset_y = pd.DataFrame(dataset_y)
    return dataset_x,dataset_y

def races_to_horses(x,y):
    x = pd.concat(x)
    y = pd.concat(y)
    x = x.reset_index(drop = True)
    y = y.reset_index(drop = True)
    return(x,y)

def add_race_info(x,race_info):
    result = []
    for i,row in enumerate(x):
        hnum = len(row.index)
        ri = race_info.iloc[i,:]
        ri_df = pd.DataFrame([ri for n in range(hnum)],columns = race_info.columns).reset_index(drop = True)
        con = pd.concat([row,ri_df],axis = 1,ignore_index = True)
        result.append(con)
    return result

def pad_race(x,y,n=18):
    result_x = []
    result_y = []
    total_len = len(x)
    count = 0
    for rx,ry in zip(x,y):
        if count%10 == 0:
            sys.stdout.write("{0}/{1}\r".format(count+1,total_len))
        sys.stdout.flush()

        length = len(rx.index)
        mean = rx.mean()
        x_add = pd.DataFrame([mean for i in range(n-length)])
        y_add = pd.DataFrame([0 for i in range(n-length)])
        cx = pd.concat([rx,x_add],ignore_index = True)
        cy = pd.concat([ry,y_add],ignore_index = True)
        result_x.append(cx)
        result_y.append(cy)
        count += 1
    print("")
    return (result_x,result_y)

def flatten_race(x):
    result = []
    for rx in x:
        row = rx.values.flatten().tolist()
        result.append(row)
    return result

def get_dummies(x,y):
    pass

def fillna_mean(dataset):
    if type(dataset) == list:
        return __fillna_mean_race(dataset)
    elif type(dataset) == pd.DataFrame:
        return __fillna_mean_horse(dataset)
    else:
        raise Exception("dataset should be list or DataFrame")

def __fillna_mean_race(ls):
    result = []
    for df in ls:
        df = __fillna_mean_horse(df)
        result.append(df)
    return result

def __fillna_mean_horse(df):
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mean())
    df = __fillna_zero_horse(df)
    return df

def fillna_zero(dataset):
    if type(dataset) == list:
        return __fillna_mean_race(dataset)
    elif type(dataset) == pd.DataFrame:
        return __fillna_mean_horse(dataset)
    else:
        raise Exception("dataset should be list or DataFrame")

def __fillna_zero_race(ls):
    result = []
    for df in ls:
        df = __fillna_zero_horse(df)
        result.append(df)
    return result

def __fillna_zero_horse(df):
    return df.fillna(0)

def under_sampling(x,y):
    con = pd.concat([y,x],axis = 1)
    lowest_frequent_value = 1
    low_frequent_records = con.ix[con.iloc[:,0] == lowest_frequent_value,:]
    other_records = con.ix[con.iloc[:,0] != lowest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.iloc[:,1:]
    con_y = con.iloc[:,0]
    return con_x,con_y

def over_sampling(x,y):
    con = pd.concat([y,x],axis = 1)
    highest_frequent_value = 1
    high_frequent_records = con.ix[con.iloc[:,0] == highest_frequent_value,:]
    other_records = con.ix[con.iloc[:,0] != highest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.iloc[:,1:]
    con_y = con.iloc[:,0]
    return con_x,con_y

def normalize(dataset):
    for df in dataset:
        pass
    mean = panel.mean()
    std  = panel.std().clip(lower=1e-4)
    panel = panel.subtract(mean,axis = 1).divide(std,axis = 1)
    return panel

if __name__ == "__main__":
    df1 = pd.DataFrame([[1,2],[3,4],[5,6],[7,None]])
    print(df1.values.flatten().tolist())
    df2 = pd.DataFrame([[1,2],[3,4],[5,6]])
    race_info = pd.DataFrame([[11,12],[13,14]])
    df = [df1,df2]
    df = add_race_info(df,race_info)

    dy1 = pd.DataFrame([0,0,1])
    dy2 = pd.DataFrame([0,0,1])
    dy = [dy1,dy2]
    x,y = pad_race(df,dy)
    x,y = races_to_horses(x,y)
