#!/usr/bin/python

import argparse
import os
import sqlite3
import sys
import reader
import feature
import util

def main():
    parser = argparse.ArgumentParser(description="generating JRDB horse race information database")
    parser.add_argument("output",nargs="?",default = "db/output.db")
    parser.add_argument("-d","--directory",default="raw_text")
    parser.add_argument("-c","--config",default="config/config.json")
    parser.add_argument("-t","--is_test",default="True")
    args = parser.parse_args()
    print(args.output)
    print(args.directory)
    print(args.config)
    print(bool(args.is_test))

    conf = util.get_config(args.config)

    is_test = args.is_test == "True"

    #create raw table
    create_db(args,is_test = is_test)

    #create feature table
    db_con = sqlite3.connect(args.output)
    reader.create_feature_table(db_con)
    db_con.close()

def create_db(args,is_test = False):
    db_con = sqlite3.connect(args.output)
    start = "031025"

    ti_orm = reader.TrainingInfoDatabase(db_con)
    csv_to_db(args,"train_info","CYB",ti_orm,test_mode = is_test,start = start)

    #process about race information
    ri_orm = reader.RaceInfoDatabase(db_con)
    csv_to_db(args,"race_info","BAC",ri_orm,test_mode = is_test,start = start)

    #process about last inforamtion
    li_orm = reader.LastInfoDatabase(db_con)
    csv_to_db(args,"last_info","TYB",li_orm,test_mode = is_test,start = start)

    #process about payoff
    pd_orm = reader.PayoffDatabase(db_con)
    csv_to_db(args,"payoff", "HJC",pd_orm,test_mode = is_test,start = start)

    #process about horse information
    hid_orm = reader.HorseInfoDatabase(db_con)
    csv_to_db(args,"horse_info","KYI",hid_orm,test_mode = is_test,start = start)

    #process about past race result
    rd_orm = reader.ResultDatabase(db_con)
    csv_to_db(args,"horse_result", "SED",rd_orm,test_mode = is_test)

    #process about extra horse information
    ex_orm = reader.ExpandedInfoDatabase(db_con)
    csv_to_db(args,"expanded_info","KKA",ex_orm,test_mode = is_test,start = start)

    #create feature table
    #reader.create_feature_table(db_con)
    #db_con.close()

def generate_dataset(args,config):
    db_con = sqlite3.connect(args.output)
    f_orm = feature.Feature(db_con)
    target_columns = config.features
    ls = [0 for i in range(18)]
    for x,y in f_orm.fetch_horse(target_columns):
        win_horse = int(x[0][0])
        ls[win_horse-1] += 1
    db_con.close()

def csv_to_db(args,dir_name,file_prefix,orm,test_mode = False,start = None,end = None):
    path = os.path.join(args.directory,dir_name)
    if not os.path.exists(path):
        raise Exception("{0} directory doesn't exist".format(dir_name))
    files = os.listdir(path)
    files = filter(lambda s:s.startswith(file_prefix),files)

    if start is not None:
        if start[0:2] != "99":
            start = "1"+start
        start_int = int(start)
        def compare_date(s):
            s = s.replace(file_prefix,"")
            s = s.replace(".txt","")
            if s[0:2] != "99":
                s = "1"+s
            return int(s) >= start_int
        files = filter(compare_date,files)

    counter = 1
    for f in files:
        if test_mode and counter > 100:
            break
        sys.stdout.write("processing : {0}/{1}\r".format(counter,len(files)))
        sys.stdout.flush()
        file_path = os.path.join(path,f)
        with open(file_path,"r") as fp:
            orm.insert_file(fp)
        counter += 1
    print("")


if __name__=="__main__":
    main()
