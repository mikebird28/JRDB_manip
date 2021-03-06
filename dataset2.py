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

def load_dataset(db_con, features, y_col = ["is_win"],limit = None, where = ""):
    x_col = ["info_race_id"] + features
    dataset_x = []
    dataset_y = []

    #db_con = sqlite3.connect(db_path)
    f_orm = feature.Feature(db_con)
    count = 0
    for x,y in f_orm.fetch_xy(x_col,y_col,where = where):
        if (limit is not None) and (count > limit):
            break
        dataset_x.append(x)
        dataset_y.append(y)
        count += 1

    dataset_x = downcast(pd.DataFrame(dataset_x,columns = x_col))
    dataset_y = downcast(pd.DataFrame(dataset_y,columns = y_col))
    return dataset_x,dataset_y

def load_x(db_con, features):
    x_col = ["info_race_id"] + features
    dataset_x = []
    f_orm = feature.Feature(db_con)
    for x in f_orm.fetch_x(x_col):
        dataset_x.append(x)
    dataset_x = downcast(pd.DataFrame(dataset_x,columns = x_col))
    return dataset_x

def nominal_columns(db_con):
    orm = reader.ColumnInfoORM(db_con)
    dic = orm.column_dict("feature")
    new_dic = {}
    for k,v in dic.items():
        if v.typ == util.NOM_SYNBOL:
            new_dic[k] = v.n
    return new_dic

def split_with_race(x,y,test_nums = 1000):
    x_col = x.columns
    y_col = y.columns
    x.reset_index(drop = True,inplace = True)
    y.reset_index(drop = True,inplace = True)

    con = pd.concat([y,x],axis = 1)
    race_id = pd.DataFrame(con["info_race_id"].unique(),columns = ["info_race_id"])
    race_id["sort_key"] = race_id["info_race_id"].apply(lambda x:x[2:])
    race_id = race_id.sort_values(by = "sort_key")["info_race_id"]

    #test_id = random.sample(race_id,test_nums)
    test_id = race_id[-test_nums:]
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
    _,i = np.unique(con.columns,return_index = True)
    con = con.iloc[:,i]

    ret_x = con.loc[:,x_col]
    ret_y = con.loc[:,y_col]
    return ret_x,ret_y

def pad_race(x,y,*additional):
    n = 18

    x_col = x.columns.tolist()
    y_col = y.columns.tolist()
    columns = x_col + y_col
    add_col = []
    for a in additional:
        col = a.columns.tolist()
        add_col.append(col)
        columns += col
 
    x.reset_index(inplace = True,drop = True)
    y.reset_index(inplace = True,drop = True)
    for a in additional:
        a.reset_index(inplace = True,drop = True)
 
    con = pd.concat([x,y]+list(additional) ,axis = 1)
    con = con.sort_values(by = "info_race_id",ascending = True)
    con = con.groupby("info_race_id").filter(lambda x:len(x) <= 18)

    size = con.groupby("info_race_id").size().reset_index(name = "counts")
    mean = con.groupby("info_race_id").mean().reset_index()
    mean.loc[:,y_col] = 0.0

    merged = mean.merge(size,on = "info_race_id",how="inner")
    for col in merged.columns:
        if col != "info_race_id" and col != "counts":
            merged[col] = merged[col].astype(np.float32)

    del mean
    del size
    #total_mean = x.mean()
    #print(total_mean)

    ls = []
    for i,row in merged.iterrows():
        pad_num = n - row["counts"]
        new_row = row[columns]
        #new_row = total_mean
        ls.extend([new_row for i in range(pad_num)])
    con = con.append(pd.DataFrame(ls))
    del ls
    con = con.sort_values(by = "info_race_id")
    ret_x = con.loc[:,x_col]
    ret_y = con.loc[:,y_col]
    ret_add = []
    for cols in add_col:
        ret_add.append(con.loc[:,cols])
    return [ret_x,ret_y] + ret_add

def pad_race_x(df,n = 18):
    x_col = df.columns.tolist()
    df = df.sort_values(by = "info_race_id",ascending = True)
    df = df.groupby("info_race_id").filter(lambda x:len(x) <= 18)

    size = df.groupby("info_race_id").size().reset_index(name = "counts")
    mean = df.groupby("info_race_id").mean().reset_index()
    target_columns = x_col

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
    return df

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
    if type(x) == pd.Series:
        x = x.to_frame()
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
    dummies = pd.DataFrame(dummies,columns = column_name)
    x = x.drop(columns,axis = 1)
    x.reset_index(drop = True,inplace = True)
    dummies.reset_index(drop = True,inplace = True)
    x = pd.concat([x,dummies],axis = 1)
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

def standardize(dataset,mx = None, mn = None, remove = []):
    if mx is None:
        mx = dataset.max()
    if mn is None:
        mn = dataset.min()
    for col in mx.index:
        if (col in remove) or (col == "info_race_id"):
            continue
        mx_c = mx[col]
        mn_c = mn[col]
        dataset[col] = (dataset[col]-mn_c)/(mx_c-mn_c)
    return dataset

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
    #for col in mean.index:
    for col in dataset.columns:
        if col in remove:
            continue
        if (not col in mean) or (not col in std):
            continue
        m = mean[col]
        s = std[col]
        dataset[col] = (dataset[col] - m)/s
    dataset = dataset.fillna(0.0)
    return dataset

def __normalize_race(dataset,mean = None,std = None):
    pass

def under_sampling(x,y,key,magnif = 1):
    if type(x) == pd.Series:
        x = x.to_frame()
    if type(y) == pd.Series:
        y = y.to_frame()

    x_col = x.columns
    y_col = y.columns
    x.reset_index(drop = True,inplace = True)
    y.reset_index(drop = True,inplace = True)
    con = pd.concat([y,x],axis = 1)
    lowest_frequent_value = 1
    low_frequent_records = con.ix[con.loc[:,key] == lowest_frequent_value,:]

    #over sampling
    low_frequent_records = pd.concat([low_frequent_records for i in range(magnif)])

    other_records = con.ix[con.loc[:,key] != lowest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))

    con = pd.concat([low_frequent_records,under_sampled_records])
    con = con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.loc[:,x_col]
    con_y = con.loc[:,y_col]
    return con_x,con_y

def over_sampling(x,y,key = "is_win"):
    if type(x) == pd.Series:
        x = x.to_frame()
    if type(y) == pd.Series:
        y = y.to_frame()

    x_col = x.columns
    y_col = y.columns
    x.reset_index(drop = True,inplace = True)
    y.reset_index(drop = True,inplace = True)
 
    con = pd.concat([y,x],axis = 1)
    highest_frequent_value = 0
    high_frequent_records = con.ix[con.loc[:,key] == highest_frequent_value,:]
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
    #y = y[target].values.tolist()
    y = y.loc[:,target]
    return (x,y)

def to_races(*args):
    for dataset in args:
        if type(dataset) == pd.DataFrame and "info_race_id" in dataset.columns:
            dataset["info_race_id"] = dataset["info_race_id"].astype(str)

    races = [[] for dataset in args]
    for dataset in args:
        dataset.reset_index(inplace = True,drop = True)

    con = pd.concat(args,axis = 1)

    #delete duplicate columns
    _,i = np.unique(con.columns,return_index = True)
    con = con.iloc[:,i]

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
        dataset.reset_index(inplace = True,drop = True)

    df = pd.concat(args,axis = 1)
    #columns = [dataset.columns for dataset in args]
    columns = []
    for dataset in args:
        c = dataset.columns
        if "info_race_id" in c:
            c = c.drop("info_race_id")
        columns.append(c)
    del args

    #delete duplicate columns
    _,i = np.unique(df.columns,return_index = True)
    df = df.iloc[:,i]

    cc = df.groupby("info_race_id").cumcount()
    df = df.set_index(["info_race_id",cc]).sort_index(1,level = 1)
    del cc

    df = df.to_panel()
    df = df.swapaxes(0,1,copy = False)
    df = df.swapaxes(1,2,copy = False)
    return [df.loc[:,:,col] for col in columns]


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

def save_cache(dataset,path):
    if not os.path.exists(path):
        os.makedirs(path)
    path_dict = {k:os.path.join(path,k) for k,v in dataset.items()}
    info_path = os.path.join(path,"cache_info")
    with open(info_path,"wb") as f:
        pickle.dump(path_dict,f)
    for k,v in path_dict.items():
        if type(dataset[k]) == list:
            with open(path_dict[k],"wb") as fp:
                pickle.dump(dataset[k],fp)
        else:
            dataset[k].to_pickle(path_dict[k])

def load_cache(path):
    dataset= {}
    info_path = os.path.join(path,"cache_info")
    with open(info_path,"rb") as fp:
        dic = pickle.load(fp)
    for k,v in dic.items():
        with open(v,"rb") as fp:
            dataset[k] = pickle.load(fp)
    return dataset

if __name__=="__main__":
    config = util.get_config("config/config.json")
    print("loading data")
    con = sqlite3.connect("db/output_v6.db")
    dx,dy = load_dataset(con,config.features,["is_win"])
    print("fill with horse mean")
    dx,dy = pad_race(dx,dy)
    dx = to_race_panel(dx)
    #fillna_mean(dx,"race")
