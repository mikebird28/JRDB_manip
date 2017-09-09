


class PaybackDatabase():
    TYPE_WIN = 1
    TYPE_PLACE = 2
    TYPE_QUINCELLA = 3
    TYPE_QUINCELLA_PLACE = 4
    TYPE_EXACTA = 5

    def __init__(self):
        pass

    def insert_file(self,fp):
        for line in fp.readlines():
            containers = []

            c.race_id = line[0:8]
            c = Container()
            c.win = line[8:10]
            c.win_payback = line[10:17]
            
            c.n1 = line[17:19]
            c.n2 = line[19:26]
            c.n3 = line[28:35]
            print(line)
            print(c)

    def __parse_line(self):
        pass

class HorseInfoDatabase():
    pass

class ResultDatabase():
    pass


class Container(dict):
    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,item):
        self.__dict__[key] = value

    def __repr__(self):
        text = ",".join(["{0}:{1}".format(k,v)for k,v in self.__dict__.items()])
        return text

if __name__=="__main__":
    pass

