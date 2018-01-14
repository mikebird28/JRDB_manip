# -*- coding : utf-8 -*- 

import dataset2
import place2vec
import pandas as pd

def load_datasets_with_p2v(db_con,features,past_n = 3):
    pre0_features = ["info_race_course_code","rinfo_discipline","rinfo_distance"]
    pre_i_features = []
    additionl_features = pre0_features
    for i in range(past_n):
        i_features = [
                "pre{0}_discipline".format(i+1),
                "pre{0}_race_course_code".format(i+1),
                "pre{0}_distance".format(i+1),
        ]
        pre_i_features.append(i_features)
        additionl_features.extend(i_features)
    x,y = dataset2.load_dataset(db_con,features + additionl_features,["is_win","win_payoff","is_place","place_payoff"])

    con = concat(x,y)
    x_col = x.columns
    y_col = y.columns

    #remove empty rows
    for c in additionl_features:
        con = con[con[c] != 0]
    con.reset_index(drop = True,inplace = True)
    x = con.loc[:,x_col]
    y = con.loc[:,y_col]
    del con

    p2v_ls = []
    p2v_0 = place2vec.get_vector(x["rinfo_discipline"],x["info_race_course_code"],x["rinfo_distance"],prefix = "pre0")

    p2v_ls.append(p2v_0)
    for i in range(past_n):
        f = [x[col] for col in pre_i_features[i]]
        prefix = "pre{0}".format(i+1)
        p2v_i = place2vec.get_vector(f[0],f[1],f[2],prefix = prefix)
        p2v_ls.append(p2v_i)

    for c in additionl_features:
        x = x.drop(c,axis = 1)

    course_info = pd.concat(p2v_ls,axis = 1)
    return (x,course_info,y)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)
