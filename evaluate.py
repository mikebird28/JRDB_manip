#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class KerasWrapper():
    def __init__(self,model):
        self.model = model

    def predict_one(self,x):
        pred = self.model.predict(x,verbose = 0)
        return pred

class KerasMultiWrapper():
    def __init__(self,model):
        self.model = model

    def predict_one(self,x):
        horse_size = x.shape[0]
        feature_size = x.shape[1]
        x = x.reshape([1,horse_size,feature_size])
        pred = self.model.predict(x,verbose = 0)
        pred = pred.reshape([horse_size])
        return pred

class KerasMultiEmbbedWrapper():
    def __init__(self,model):
        self.model = model

    def predict_one(self,x):
        pred = self.model.predict(x,verbose = 0)
        return pred

class ScikitWrapper():
    def __init__(self, model):
        self.model = model

    def predict_one(self,x):
        pred = self.model.predict_proba(x)[:,1].ravel()
        return pred

class RegressionWrapper():
    def __init__(self, model):
        self.model = model

    def predict_one(self,x):
        pass
def top_n_k(wrapper,race_x,race_y,payoff,n = 1):
    buy_num = 0
    race_num = 0
    correct = 0
    rewards = 0

    for x,y,p in zip(race_x,race_y,payoff):
        pred = wrapper.predict_one(x)
        binary_pred = to_descrete(pred,top_n = n)

        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,binary_pred)
        ret = np.dot(p,binary_pred)
        if c > 0:
            correct += c
            rewards += ret
        race_num += 1
        buy_num += np.sum(binary_pred)
    return (float(correct)/race_num,float(rewards)/buy_num)

def top_n_k_remove_first(model,race_x,race_y,payoff,odds,n = 1):
    buy_num = 0
    race_num = 0
    correct = 0
    rewards = 0

    for x,y,p,odds in zip(race_x,race_y,payoff,odds):
        odds = odds.loc[:,"linfo_win_odds"].values
        pred = model.predict_one(x)
        first_pop = 1 - np.clip(to_descrete(odds,mode = "min",top_n = 1),0,1)
        binary_pred = first_pop * to_descrete(pred,top_n = n)

        bn = np.sum(binary_pred)

        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,binary_pred)
        ret = np.dot(p,binary_pred)
        if c > 0:
            correct += c
            rewards += ret
        race_num += 1
        buy_num += bn
    return (buy_num,float(correct)/buy_num,float(rewards)/buy_num)

def to_descrete(array,mode = "max",top_n = 1):
    res = np.zeros_like(array)
    if mode == "min":
        rank = array.argsort()
        res = np.zeros_like(rank)
        rank = rank[:top_n]
        res[rank] = 1
    elif mode == "max":
        rank = array.argsort()[::-1]
        res = np.zeros_like(rank)
        rank = rank[:top_n]
        res[rank] = 1
    return res

def plot_importance(column,importance):
    x = range(len(column))
    plt.barh(x,importance)
    plt.yticks(x,column)
    plt.show()
    return

def show_importance(column,importance,sort = True,top_n = None):
    zipped = zip(column,importance)
    if sort:
        zipped = sorted(zipped,key = lambda x : x[1],reverse = True)

    counter = 0
    for f,i in zipped:
        if (top_n is not None) and (not counter < 20):
            break
        print("{0:<25} : {1:.5f}".format(f,i))
        counter += 1

def show_similarity(target_idx,vectors):
    columns_name = []
    dic_1 = [u"芝",u"ダート",u"障害"]
    dic_2 = [u"短距離",u"短中距離",u"中距離",u"中長距離",u"長距離"]
    for i in range(165):
        dist = i//33
        course = (i%33)//3
        typ = (i%33)%3
        columns_name.append(u"{0}-{1:<3}-{2:<4}".format(course,dic_1[typ],dic_2[dist]))
    results = []
    for i in range(len(vectors)):
        cs = cos_similarity(vectors[target_idx],vectors[i])
        results.append((i,cs))
    results = sorted(results,key = lambda x: x[1])
    for t in results:
        print(u"{0} : {1}".format(columns_name[t[0]],t[1]))

def cos_similarity(x1,x2):
    cs = np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
    return cs


if __name__ =="__main__":
    array = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(array)
    show_similarity(0,array)
    pass

