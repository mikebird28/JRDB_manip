#!/usr/bin/python

import argparse
import os
import sqlite3
import reader

def main():
    parser = argparse.ArgumentParser(description="generating JRDB horse race information database")
    parser.add_argument("output",nargs="?",default = "output.db")
    parser.add_argument("-d","--directory",default="raw_text")
    args = parser.parse_args()
    print(args.output)
    print(args.directory)

    db_con = sqlite3.connect(args.output)
    IS_TEST = True

    "process about horse information"
    hid_orm = reader.HorseInfoDatabase(db_con)
    csv_to_db(args,"horse_info","KYI",hid_orm,test_mode = IS_TEST)

    "process about race result"
    rd_orm = reader.ResultDatabase(db_con)
    csv_to_db(args,"horse_result", "SED",rd_orm,test_mode = IS_TEST)

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
        print("processing : {0}/{1}".format(counter,len(files)))
        file_path = os.path.join(path,f)
        with open(file_path,"r") as fp:
            orm.insert_file(fp)
        counter += 1


if __name__=="__main__":
    main()
