#-*- coding:utf-8 -*-


#functions which convert data to appropriate type
def to_integer(x):
    try:
        return int(x)
    except ValueError:
        return None

def to_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def to_string(x):
    return x.strip()

def to_unicode(x):
    return x.decode("shift-jis").strip()

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
    def __init__(self):
        pass

    def insert_file(self,fp):
        for line in fp.readlines():
            c = Container()
            c.race_id = line[0:8]
            c.horse_number = line[8:10]
            c.pedigree_id = line[10:18]
            c.registered_id = line[18:26]

class ResultDatabase():
    def __init__(self):
        pass

    def insert_file(self,fp):
        for line in fp.readlines():
            c = Container()
            c.race_id              = to_string(line[0:8]) #レースキー
            c.horse_number         = to_integer(line[8:10]) #馬番
            c.pedigree_id          = to_string(line[10:18]) #血統登録番号
            c.registered_date      = to_string(line[18:26]) #登録日
            c.horse_name           = to_unicode(line[26:62]) #名前
            c.distance             = to_integer(line[62:66]) #距離
            c.discipline           = to_integer(line[66]) #芝ダ障害コード
            c.left_or_right        = to_integer(line[67])#右左
            c.in_or_out            = to_integer(line[68])#内外
            c.field_status         = to_integer(line[69:71])#馬場状態

            c.race_category        = to_integer(line[71:73])#種別
            c.race_condition       = to_string(line[73:75])#条件
            c.race_remarks         = to_integer(line[75:78])#記号
            c.race_weights         = to_integer(line[78])#重量
            c.race_grade           = to_integer(line[79])#グレード
            c.race_name            = to_unicode(line[80:130])#レース名
            c.race_headcount       = to_integer(line[130:132])#頭数
            c.race_alias           = to_unicode(line[132:140])#レース名略称

            c.order_of_finish      = line[140:142] # 着順
            c.irregular_category   = line[142] #異常区分
            c.finishing_time       = line[143:147] #タイム
            c.basis_weight         = line[147:150] #斤量
            c.jockey_name          = to_unicode(line[150:162]) #騎手名
            c.trainer_name         = to_unicode(line[162:174]) #調教師名
            c.odds                 = to_float(line[174:180]) #確定オッズ
            c.popularity           = line[180:182] #人気順位

            c.jrdb_idm             = line[182:185]
            c.jrdb_raw_score       = line[185:188]
            c.jrdb_field_info      = line[188:191]
            c.jrdb_pace_info       = line[191:194]
            c.late_start           = line[194:197]
            c.position             = line[197:200]
            c.disadvantage         = line[200:203]
            c.disadvantage_opening = line[203:206]
            c.disadvantage_middle  = line[206:209]
            c.disadvantage_final   = line[209:212]
            c.race_info            = line[212:215]
            c.course_position      = line[215]
            c.condition_code       = line[216]
            c.class_code           = line[217:219]
            c.body_code            = line[219]
            c.atmosphere_code      = line[220]
            c.race_pace            = line[221]
            c.horse_pace           = line[222]
            c.firstphase_score     = line[223:228]
            c.lastphase_score      = line[228:233]
            c.pase_score           = line[233:238]
            c.race_pace_score      = line[238:243]
            c.first_horse_name     = to_unicode(line[243:255])
            c.time_delta           = line[255:258]
            c.first_3f_time        = line[258:261]
            c.last_3f_time         = line[261:264]
            c.jrdb_remarks         = line[264:288]

            c.place_odds           = line[290:296]
            c.morning_odds         = line[296:302]
            c.morning_place_odds   = line[302:308]
            c.pass_1               = line[308:310]
            c.pass_2               = line[310:312]
            c.pass_3               = line[312:314]
            c.pass_4               = line[314:316]
            c.first_3f_delta       = line[316:319]
            c.last_3f_delta        = line[319:322]
            c.jockey_id            = line[322:327]
            c.trainer_id           = line[327:332]
            c.weight               = line[332:335]
            c.weight_delta         = line[335:338]
            c.weather_code         = line[338:339]
            c.course_info          = line[339]
            c.race_running_style   = line[340]

            c.payback_win          = line[341:348]
            c.payback_place        = line[348:355]
            c.prize                = line[355:360]
            c.class_prize          = line[360:365]
            c.position_at_corner   = line[369]
            c.show()


class Container(dict):
    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,item):
        self.__dict__[key] = value

    def __repr__(self):
        text = "Container : {0} values".format(len(self.__dict__))
        return text

    def show(self):
        text = u",\n".join([u"{0:<20}:{1}".format(k,v) for k,v in self.__dict__.items()])
        #text = "HEL"
        print(text)
        

if __name__=="__main__":
    pass

