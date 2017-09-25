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
    parser.add_argument("output",nargs="?",default = "output.db")
    parser.add_argument("-d","--directory",default="raw_text")
    parser.add_argument("-c","--config",default="config.json")
    args = parser.parse_args()
    print(args.output)
    print(args.directory)
    print(args.config)

    conf = util.get_config(args.config)

    create_db(args,is_test = True)
    #generate_dataset(args,conf)

def create_db(args,is_test = False):
    db_con = sqlite3.connect(args.output)

    #process about payoff
    pd_orm = reader.PayoffDatabase(db_con)
    csv_to_db(args,"payoff", "HJC",pd_orm,test_mode = is_test)

    #process about horse information
    hid_orm = reader.HorseInfoDatabase(db_con)
    csv_to_db(args,"horse_info","KYI",hid_orm,test_mode = is_test)

    #process about past race result
    rd_orm = reader.ResultDatabase(db_con)
    csv_to_db(args,"horse_result", "SED",rd_orm,test_mode = is_test)

    ex_orm = reader.ExpandedInfoDatabase(db_con)
    csv_to_db(args,"expanded_info","KKA",ex_orm,test_mode = is_test)


    #create feature table
    reader.create_feature_table(db_con)
    db_con.close()

def generate_dataset(args,config):
    db_con = sqlite3.connect(args.output)
    f_orm = feature.Feature(db_con)
    target_columns = config.features
    ls = [0 for i in range(18)]
    for x,y in f_orm.fetch_horse(target_columns):
        win_horse = int(x[0][0])
        ls[win_horse-1] += 1
    print(ls)
        
    db_con.close()

def csv_to_db(args,dir_name,file_prefix,orm,test_mode = False):
    path = os.path.join(args.directory,dir_name)
    if not os.path.exists(path):
        raise Exception("{0} directory doesn't exist".format(dir_name))
    files = os.listdir(path)
    files = filter(lambda s:s.startswith(file_prefix),files)
    counter = 1
    for f in files:
        if test_mode and counter > 10:
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
