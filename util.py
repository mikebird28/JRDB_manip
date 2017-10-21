#-*- coding:utf-8 -*-

import json

STR_SYNBOL = "STR"
UNI_SYNBOL = "UNI"
INT_SYNBOL = "INT"
FLO_SYNBOL = "FLO"
NOM_SYNBOL = "NOM"

class Container(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        text = "Container : {0} values".format(len(self.__dict__))
        return text

    def show(self):
        text = u",\n".join([u"{0:<20}:{1}".format(k,v.value) for k,v in self.items()])
        print(text)

class Maybe():
    def __init__(self,typ,value):
        self.typ = typ
        self.value = value

    def string(self):
        str_op = lambda x : u"'"+ x + u"'"
        nominal_op = lambda x : u"'" + unicode(x[0]) + u"'" 
        func_dict = {STR_SYNBOL:str_op,UNI_SYNBOL:str_op,INT_SYNBOL:str,FLO_SYNBOL:str,NOM_SYNBOL:nominal_op}
        if self.value != None:
            string = func_dict[self.typ](self.value)
        else:
            string = ""
        return string

class Nominal:
    pass


class Config(object):
    def __init__(self,js):
        self.config = js["config"]
        self.features = js["features"]
        self.features_light = js["features_light"]

def get_config(path):
    fp = open(path,"r")
    js = json.load(fp)
    return Config(js)

if __name__=="__main__":
    print(get_config("./config.json").features)
