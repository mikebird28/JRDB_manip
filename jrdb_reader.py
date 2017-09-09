#!/usr/bin/python

import argparse
import os
import reader

def main():
    parser = argparse.ArgumentParser(description="generating JRDB horse race information database")
    parser.add_argument("output",nargs="?",default = "output.db")
    parser.add_argument("-d","--directory",default="raw_text")
    args = parser.parse_args()
    print(args.output)
    print(args.directory)

    "process about horse information"
    info_path = os.path.join(args.directory,"horse_info")
    if not os.path.exists(info_path):
        raise Exception("horse_info directory doesn't exist")
    files = os.listdir(info_path)
    files = filter(lambda s:s.startswith("KYI"),files)

    "process about race result"
    result_path = os.path.join(args.directory,"horse_result")
    if not os.path.exists(result_path):
        raise Exception("horse_result directory doesn't exist")
    files = os.listdir(result_path)
    files = filter(lambda s:s.startswith("SED"),files)


    "process about payback"
    for f in files:
        break
        path = os.path.join(args.directory,f)
        with open(path,"r") as fp:
            pr = reader.PaybackDatabase()
            pr.insert_file(fp)


if __name__=="__main__":
    main()
