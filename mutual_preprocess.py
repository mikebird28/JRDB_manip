# -*- coding : utf-8 -*- 

import dataset2
import place2vec
import pandas as pd

def load_datasets_with_p2v(db_con,features):
    pre0_features = ["info_race_course_code","rinfo_discipline","rinfo_distance"]
    pre1_features = ["pre1_race_course_code","pre1_discipline","pre1_distance",]
    pre2_features = ["pre2_race_course_code","pre2_discipline","pre2_distance"]
    pre3_features = ["pre3_race_course_code","pre3_discipline","pre3_distance"]
    additionl_features = pre0_features + pre1_features + pre2_features + pre3_features
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

    p2v_0 = place2vec.get_vector(x["rinfo_discipline"],x["info_race_course_code"],x["rinfo_distance"],prefix = "pre0")
    p2v_1 = place2vec.get_vector(x["pre1_discipline"],x["pre1_race_course_code"],x["pre1_distance"],prefix = "pre1")
    p2v_2 = place2vec.get_vector(x["pre2_discipline"],x["pre2_race_course_code"],x["pre2_distance"],prefix = "pre2")
    p2v_3 = place2vec.get_vector(x["pre3_discipline"],x["pre3_race_course_code"],x["pre3_distance"],prefix = "pre3")
    for c in additionl_features:
        x = x.drop(c,axis = 1)

    course_info = pd.concat([p2v_0,p2v_1,p2v_2,p2v_3],axis = 1)
    return (x,course_info,y)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)
