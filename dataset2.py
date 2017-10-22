import sqlite3
import os
import sys
import pandas as pd
import random
import numpy as np
import feature
import util
import reader
import pickle
from sklearn.preprocessing import OneHotEncoder

def load_dataset(db_path, features, y_col = ["is_win"]):
    x_col = ["info_race_id"] + features
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect(db_path)
    f_orm = feature.Feature(db_con)
    for x,y in f_orm.fetch_with_xy(x_col,y_col):
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = downcast(pd.DataFrame(dataset_x,columns = x_col))
    dataset_y = downcast(pd.DataFrame(dataset_y,columns = y_col))
    return dataset_x,dataset_y

def nominal_columns(db_path):
    con = sqlite3.connect(db_path)
    orm = reader.ColumnInfoORM(con)
    dic = orm.column_dict("feature")
    new_dic = {}
    for k,v in dic.items():
        if v.typ == util.NOM_SYNBOL:
            new_dic[k] = v.n
    return new_dic

def split_with_race(x,y,test_nums = 1000):
    x_col = x.columns
    y_col = y.columns

    con = pd.concat([y,x],axis = 1)
    race_id = con["info_race_id"].unique()
    test_id = random.sample(race_id,test_nums)
    test_con = con[con["info_race_id"].isin(test_id)]
    test_x = test_con.loc[:,x_col]
    test_y = test_con.loc[:,y_col]
    train_con = con[~con["info_race_id"].isin(test_id)]
    train_x = train_con.loc[:,x_col]
    train_y = train_con.loc[:,y_col]
    return train_x,test_x,train_y,test_y

def add_race_info(x,y,race_info):
    x_col = pd.Index(x.columns.tolist() + race_info.columns.tolist()).unique()
    y_col = y.columns
    con = pd.concat([x,y],axis = 1)
    con = con.merge(race_info,on = "info_race_id",how = "left")
    #con = con.loc[:,~con.columns.duplicated()]
    _,i = np.unique(con.columns,return_index = True)
    con = con.iloc[:,i]

    ret_x = con.loc[:,x_col]
    ret_y = con.loc[:,y_col]
    return ret_x,ret_y

def pad_race(x,y,n=18):
    x_col = x.columns.tolist()
    y_col = y.columns.tolist()
    columns = x_col + y_col
    df = pd.concat([y,x],axis = 1)
    df = df.sort_values(by = "info_race_id",ascending = True)
    df = df.groupby("info_race_id").filter(lambda x:len(x) <= 18)

    size = df.groupby("info_race_id").size().reset_index(name = "counts")
    mean = df.groupby("info_race_id").mean().reset_index()
    mean.loc[:,y.columns] = 0
    target_columns = columns

    merged = mean.merge(size,on = "info_race_id",how="inner")
    ls = []
    for i,row in merged.iterrows():
        rid = row["info_race_id"]
        hnum = row["counts"]
        pad_num = n - row["counts"]
        new_row = row[target_columns]
        for i in range(pad_num):
            ls.append(new_row)
    pad_data = pd.DataFrame(ls,columns = target_columns)
    df = df.append(pad_data)
    df = df.sort_values(by = "info_race_id")
    size = df.groupby("info_race_id").size().reset_index(name="counts")
    dfx = df.loc[:,x_col]
    dfy = df.loc[:,y_col]
    return (dfx,dfy)

def flatten_race(df):
    ls = []
    rids = []
    count = 0
    x_col = df.columns
    x_col = x_col.drop("info_race_id")
    groups = df.groupby("info_race_id")
    for rid ,g in groups:
        count += 1
        if count%100 == 0:
            print(count)
        flatten = g[x_col].values[0:18].flatten().tolist()
        #flatten = g[x_col].values.flatten().tolist()
        ls.append(flatten)
        rids.append(rid)
    dfx = pd.DataFrame(ls)
    dfid = pd.DataFrame(rids,columns=["info_race_id"])
    df = pd.concat([dfx,dfid],axis = 1)
    return df

def flatten_race2(df):
    ls = []
    rids = []
    count = 0
    x_col = df.columns
    x_col = x_col.drop("info_race_id")
    cc = df.groupby("info_race_id").cumcount() + 1
    cc.name = "__cc"
    df = pd.concat([cc,df],axis = 1)
    df = df[df["__cc"] <= 18]
    df = df.set_index(["info_race_id","__cc"])
    df = df.unstack().sort_index(1,level = 1)
    df.columns = ['_'.join(map(str,i)) for i in df.columns]
    df = df.reset_index()
    return df


def get_dummies(x,col_dic):
    pairs = {}
    for col in x.columns:
        try:
            pairs[col] = col_dic[col] + 1
        except KeyError:
            pass
    pairs = pairs.items()
    if len(pairs) == 0:
        return x
    pairs = sorted(pairs)
    columns = map(lambda x:x[0], pairs)
    n_values = map(lambda x:x[1], pairs)
    column_name = []
    for k,v in pairs:
        cols = ["{0}_{1}".format(k,i) for i in range(v)]
        column_name.extend(cols)

    ohe = OneHotEncoder(n_values = n_values,sparse = False)
    x.loc[:,columns] = x.loc[:,columns].fillna(0)
    tmp_x = x.loc[:,columns]

    ohe.fit(tmp_x)
    dummies = ohe.transform(tmp_x).astype(np.int8)
    dummies = pd.DataFrame(dummies,index = tmp_x.index,columns = column_name)
    x = x.drop(columns,axis = 1)
    x = x.merge(dummies,left_index = True,right_index = True)
    return x

def dummy_column(x,col_dic):
    pairs = {}
    for col in x.columns:
        try:
            pairs[col] = col_dic[col] + 1
        except KeyError:
            pass
    pairs = pairs.items()
    if len(pairs) == 0:
        return x
    pairs = sorted(pairs)
    columns = map(lambda x:x[0], pairs)
    n_values = map(lambda x:x[1], pairs)
    column_name = []
    for k,v in pairs:
        cols = ["{0}_{1}".format(k,i) for i in range(v)]
        column_name.extend(cols)
    return column_name

def fillna_mean(dataset,typ = "horse"):
    if typ == "horse":
        return __fillna_mean_horse(dataset)
    elif typ == "race":
        return __fillna_mean_race(dataset)
    else:
        raise Exception("unknown separate type")

def __fillna_mean_race(df):
    means_stds = df.groupby("info_race_id").agg(["mean"]).reset_index()
    columns = df.columns
    for col in columns:
        if col is "info_race_id":
            continue
        ms = means_stds[col]
        df = df.join(ms,on = "info_race_id")
        df = df.fillna(df["mean"])
        df = df.drop("mean",axis=1)
    return df

def __fillna_mean_horse(df):
    mean = df.mean(numeric_only = True)
    df = df.fillna(mean)
    return df

def fillna_zero(df):
    columns = df.columns
    for col in columns:
        try:
            df = df.fillna(0)
        except Exception:
            pass
    return df

def normalize(dataset,typ="horse",mean = None,std = None,remove = []):
    if typ == "horse":
        return __normalize_horse(dataset,mean,std,remove)
    elif typ == "race":
        return __normalize_race(dataset)
    else:
        raise Exception("unknown separate type")

def __normalize_horse(dataset,mean = None,std = None,remove = []):
    if type(mean) == type(None):
        mean = dataset.mean(numeric_only = True)
    if type(std) == type(None):
        std = dataset.std(numeric_only = True).clip(1e-3)
    dataset = dataset.copy()
    for col in mean.index:
        if col in remove:
            continue
        m = mean[col]
        s = std[col]
        dataset[col] = (dataset[col] - m)/s
    return dataset

def __normalize_race(dataset,mean = None,std = None):
    pass

def under_sampling(x,y,key = "is_win"):
    x_col = x.columns
    y_col = y.columns
    con = pd.concat([y,x],axis = 1)
    lowest_frequent_value = 1
    low_frequent_records = con.ix[con.loc[:,key] == lowest_frequent_value,:]
    other_records = con.ix[con.loc[:,key] != lowest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.loc[:,x_col]
    con_y = con.loc[:,y_col]
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

def for_use(x,y,target):
    if "info_race_id" in x.columns:
        x = x.drop("info_race_id",axis = 1)
    y = y[target].values.tolist()
    return (x,y)

def to_races(*args):
    for dataset in args:
        if type(dataset) == pd.DataFrame and "info_race_id" in dataset.columns:
            dataset["info_race_id"] = dataset["info_race_id"].astype(str)

    races = [[] for dataset in args]
    #races = [np.zeros(len(dataset),18,columns)]
    con = pd.concat(args,axis = 1)

    #delete duplicate columns
    _,i = np.unique(con.columns,return_index = True)
    con = con.iloc[:,i]

    """
    cc = con.groupby("info_race_id").cumcount() 
    con = con.set_index(["info_race_id",cc])
    print(con)
    print(con.axes)
    grouped = con.groupby("info_race_id")
    print("Panel")
    panel = pd.Panel(grouped.groups)
    print(panel)
    """

    grouped = con.groupby("info_race_id")
    for name,group in grouped:
        for i,dataset in enumerate(args):
            if type(dataset) == pd.Series:
                columns = dataset.name
            elif type(dataset) == pd.DataFrame:
                columns = dataset.columns
                columns = columns.drop("info_race_id",errors = "ignore")
            else:
                raise Exception("Dataset which has unknown type was passed")
            #race = group[columns].as_matrix()
            race = group[columns]
            races[i].append(race)
    return races

def to_race_panel(*args):
    for dataset in args:
        if type(dataset) == pd.DataFrame and "info_race_id" in dataset.columns:
            dataset["info_race_id"] = dataset["info_race_id"].astype(str)

    races = [[] for dataset in args]
    con = pd.concat(args,axis = 1)

    #delete duplicate columns
    _,i = np.unique(con.columns,return_index = True)
    con = con.iloc[:,i]

    cc = con.groupby("info_race_id").cumcount() 
    con = con.set_index(["info_race_id",cc]).sort_index(1,level = 1)
    panel = con.to_panel()
    panel = panel.astype(np.float32)
    print(panel.dtypes)
    panel = panel.swapaxes(0,1,copy = False)
    panel = panel.swapaxes(1,2,copy = False)

    results = []
    for dataset in args:
        p = panel.loc[:,:,dataset.columns]
        results.append(p)
    return results


def races_to_numpy(dataset):
    new_races = []
    for race in dataset:
        if type(race) == pd.Series:
            new_race = race.values.tolist()
        elif type(race) == pd.DataFrame:
            new_race = race.as_matrix()
        new_races.append(new_race)
    return new_races

def downcast(dataset):
    for clmns in dataset.columns:
        if dataset[clmns].dtype == np.float64:
            dataset[clmns] = pd.to_numeric(dataset[clmns], downcast = "float")
        if dataset[clmns].dtype == np.int64:
            dataset[clmns] = pd.to_numeric(dataset[clmns], downcast = "integer")
    return dataset

def save_cache(dir_path,dataset):
    path_dict = {k:os.path.join(dir_path,k) for k,v in dataset.items()}
    info_path = os.path.join(dir_path,"cache_info")
    with open(info_path,"wb") as f:
        pickle.dump(path_dict,f)


def load_cache(path):
    pass


if __name__=="__main__":
    config = util.get_config("config/config.json")
    print("loading data")
    dx,dy = load_dataset("db/output_v6.db",config.features)
    print("fill with horse mean")
    dx,dy = pad_race(dx,dy)
    dx = flatten_race2(dx)
    #fillna_mean(dx,"race")
