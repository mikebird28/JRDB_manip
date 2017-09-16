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

    "process about horse information"
    info_path = os.path.join(args.directory,"horse_info")
    if not os.path.exists(info_path):
        raise Exception("horse_info directory doesn't exist")
    files = os.listdir(info_path)
    files = filter(lambda s:s.startswith("KYI"),files)
    counter = 1
    hfd = reader.HorseInfoDatabase(db_con)
    for f in files:
        print("processing : {0}/{1}".format(counter,len(files)))
        path = os.path.join(info_path,f)
        with open(path,"r") as fp:
            hfd.insert_file(fp)
        counter += 1
        break

    "process about race result"
    result_path = os.path.join(args.directory,"horse_result")
    if not os.path.exists(result_path):
        raise Exception("horse_result directory doesn't exist")
    files = os.listdir(result_path)
    files = filter(lambda s:s.startswith("SED"),files)
    counter = 1
    rd = reader.ResultDatabase(db_con)
    for f in files:
        print("processing : {0}/{1}".format(counter,len(files)))
        path = os.path.join(result_path,f)
        with open(path,"r") as fp:
            rd.insert_file(fp)
        counter += 1
        break

    "process about payback"
    for f in files:
        break
        path = os.path.join(args.directory,f)
        with open(path,"r") as fp:
            pr = reader.PaybackDatabase()
            pr.insert_file(fp)
    db_con.close()


if __name__=="__main__":
    main()
