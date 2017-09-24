
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
