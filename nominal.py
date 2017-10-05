
def nominal_int(x):
    try:
        ret = int(x)
        return ret
    except:
        return 0

def nominal_pace(x):
    if x == "H":
        return 1
    if x == "M":
        return 2
    elif x == "S":
        return 3
    else:
        return 0

def nominal_field_status(x):
    dic = {
        "10":1,
        "11":2,
        "12":3,
        "20":4,
        "21":5,
        "22":6,
        "30":7,
        "31":8,
        "32":9,
        "40":10,
        "41":11,
        "41":12,
    }
    try:
        return dic[x]
    except KeyError:
        return 0

def nominal_paddok(x):
    dic = {
        "A":1,
        "B":2,
        "C":3,
        "D":4,
        "E":5
    }
    try:
        return dic[x]
    except KeyError:
        return 0

def nominal_condiction_score(x):
    dic = {
        "AA":1,
        "A":2,
        "B":3,
        "C":4
    }
    try:
        return dic[x]
    except KeyError:
        return 0

def nominal_course_info(x):
    dic = {
        "AA":1,
        "A":2,
        "B":3,
        "C":4
    }
    try:
        return dic[x]
    except KeyError:
        return 0

def nominal_running_style(x):
    dic = {
        "AA":1,
        "A":2,
        "B":3,
        "C":4
    }
    try:
        return dic[x]
    except KeyError:
        return 0
