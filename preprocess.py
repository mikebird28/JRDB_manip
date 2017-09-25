-*- coding:utf-8 -*-

def race_to_horses(x,y):
    result_x = []
    result_y = []
    for rx,ry in zip(x,y):
        for hx,hy in zip(rx,ry):
            result_x.append(hx)
            result_y.append(hy)
    return (result_x,result_y)

def pad_race(x,y,n = 18,columns_dict = {}):
    for rx,ry in zip(x,y):
        for i in range(N):
            pass
