#-*- coding:utf-8 -*-


#functions which convert data to appropriate type
def to_integer(x,illegal_value = None):
    """
    convert x to integer
    if x is not integer, return None

    Args:
        x : variable which you want to convert to integer 
        illegal_value : value which use when x is not convetable

    """
    try:
        return Maybe(int,int(x))
    except ValueError:
        return Maybe(int,None)

def to_float(x,illegal_value = None):
    try:
        return Maybe(float,float(x))
    except ValueError:
        return Maybe(float,illegal_value)

def to_string(x,illegal_value = None):
    text = x.strip()
    if text == "":
        return Maybe(str,None)
    else:
        return Maybe(str,text)

def to_unicode(x,illegal_value = None):
    text = x.decode("cp932").strip()
    if text == u"":
        return Maybe(unicode,None)
    else:
        return Maybe(unicode,text)

class Maybe():
    def __init__(self,typ,value):
        self.typ = typ
        self.value = value

    def string(self):
        str_op = lambda x : u"'"+ x + u"'"
        func_dict = {str:str_op,unicode:str_op,int:str,float:str}
        if self.value != None:
            string = func_dict[self.typ](self.value)
        else:
            string = ""
        return string

#utility functions for database access
def table_exists(con,table_name):
    """check if the table has already existed
    Args:
        con       : database connection
        tabe_name      : table name
    Returns:
        return True if transaction was successed
        return False if transaction was failed
    """
 
    c = con.cursor()
    sql = """select count(*) from sqlite_master where type='table' and name='{0}'""".format(table_name)
    c.execute(sql)
    result = c.fetchone()[0]
    if result == 1:
        return True
    else:
        return False

def create_table(con,name,type_dict):
    """create new tables in the given connectio
    Args:
        con       : database connection
        name      : table name
        type_dict : dictionary of columns and type
    Returns:
        return True if transaction was successed
        return False if transaction was failed
    """
    type_to_str = {str:"TEXT",unicode:"TEXT",int:"INTEGER",float:"REAL"}
    type_tuples = []
    keys = type_dict.keys()
    keys.sort()
    for k in keys:
        ctype = type_dict[k]
        type_str = type_to_str[ctype]
        type_tuples.append((k,type_str))
    columns = ",".join(["{0} {1}".format(k,v) for k,v in type_tuples])
    sql = """CREATE TABLE {name}({columns});""".format(name = name,columns = columns)
    c = con.cursor()
    c.execute(sql)

def insert_container(con,name,containers):
    """insert container to the table
    Args:
        con       : database connection
        name      : table name
        container : container whish has horse race records
    Returns:
        return True if transaction was successed
        return False if transaction was failed
    """
    keys = []
    values = []
    for k,v in containers.items():
        if v.value == None:
            continue
        keys.append(k)
        values.append(v.string())

    keys_text = u",".join(keys)
    values_text = u",".join(values)
    sql = u"insert into {0}({1}) values({2});".format(name,keys_text,values_text)
    con.execute(sql)

def inference_type(containers):
    """ inference container's attribute type
    Args:
        containers : list of Container
    Returns:
        dictionary of container's attribute type
    """
    type_dict = {}
    is_first = True
    bef_c = None
    for c in containers:
        if is_first:
            bef_c = c
            type_dict = {k:bef_c.get(k).typ for k in bef_c.keys()}
            continue
        for k in c.keys():
            if c.get(k).typ != bef_c.get(k).typ:
                raise Exception("Container attribute's type is something wrong")
    return type_dict


class BaseORM(object):
    def __init__(self,con,table_name):
        self.table_name = table_name
        self.con = con

    def insert_file(self,fp):
        containers = []
        for line in fp.readlines():
            c = self.parse_line(line)
            containers.append(c)
        if not table_exists(self.con,self.table_name):
            type_dict = inference_type(containers)
            create_table(self.con,self.table_name,type_dict)
        for c in containers:
            insert_container(self.con,self.table_name,c)
        self.con.commit()
 

class PaybackDatabase(BaseORM):
    def __init__(self,con):
        super(PaybackDatabase,self).__init__(con,"payback")

    def __parse_line(self):
        c = Container()
        c.race_id = line[0:8]

class HorseInfoDatabase(BaseORM):
    def __init__(self,con):
        super(HorseInfoDatabase,self).__init__(con,"horse_info")

    def parse_line(self,line):
        c = Container()
        c.race_id              = to_string(line[0:8])    #レースキー
        c.horse_number         = to_integer(line[8:10])  #馬番
        c.pedigree_id          = to_string(line[10:18])  #血統登録番号
        c.horse_name           = to_unicode(line[18:54]) #名前

        c.idm                  = to_float(line[54:59])   #IDM
        c.jockey_score         = to_float(line[59:64])   #騎手指数
        c.info_score           = to_float(line[64:69])   #情報指数
        c.yobi_1               = to_float(line[69:74])   #予備
        c.yobi_2               = to_float(line[74:79])   #予備
        c.yobi_3               = to_float(line[79:84])   #予備
        c.composite_score      = to_float(line[84:89])   #総合指数

        c.running_style        = to_integer(line[89])    #脚質
        c.distance_fitness     = to_integer(line[90])    #距離適性
        c.condiction_score     = to_integer(line[91])    #上昇度
        c.rotation             = to_integer(line[92:95]) #ローテーション

        c.base_odds            = to_float(line[95:100])  #基準オッズ
        c.base_popularity      = to_integer(line[100:102]) #基準人気順位
        c.base_place_odds      = to_float(line[102:107])   #基準複勝オッズ
        c.base_place_popularity = to_integer(line[107:109])#基準複勝人気順位
        c.specific_info_5      = to_integer(line[109:112]) #特定情報◎
        c.specific_info_4      = to_integer(line[112:115]) #特定情報○
        c.specific_info_3      = to_integer(line[115:118]) #特定情報▲
        c.specific_info_2      = to_integer(line[118:121]) #特定情報△
        c.specific_info_1      = to_integer(line[121:124]) #特定情報×
        c.general_info_5       = to_integer(line[124:127]) #総合情報◎
        c.general_info_4       = to_integer(line[127:130]) #総合指数○
        c.general_info_3       = to_integer(line[130:133]) #総合指数▲
        c.general_info_2       = to_integer(line[133:136]) #総合指数△
        c.general_info_1       = to_integer(line[136:139]) #総合指数×
        c.popurality_score     = to_integer(line[139:144]) #人気指数
        c.training_score       = to_integer(line[144:149]) #調教師数
        c.stable_score         = to_integer(line[149:154]) #厩舎指数

        c.training_sign_code   = to_integer(line[154])     #調教矢印コード
        c.stable_eval_code     = to_integer(line[155])     #厩舎評価コード
        c.jockey_quinella      = to_float(line[156:160])   #騎手期待値連対率
        c.running_score        = to_integer(line[160:163]) #激走指数
        c.hoof_code            = to_integer(line[163:165]) #蹄コード
        c.heavy_fitness_code   = to_integer(line[165])     #重適正コード
        c.class_code           = to_integer(line[166:168]) #クラスコード
        c.yobi_4               = to_integer(line[168:170])     #予備

        c.brinker              = to_integer(line[170])     #ブリンカー
        c.jockey_name          = to_unicode(line[171:183]) #騎手名
        c.basis_weight         = to_integer(line[183:186]) #負担重量
        c.apprentice_class     = to_integer(line[186])     #見習い区分
        c.trainer_name         = to_unicode(line[187:199]) #調教師名
        c.trainer_division     = to_unicode(line[199:203]) #調教師所属

        c.pre1_result_id       = to_string(line[203:219])  #前走1競争成績キー
        c.pre2_result_id       = to_string(line[219:235])  #前走2競争成績キー
        c.pre3_result_id       = to_string(line[235:251])  #前走3競争成績キー
        c.pre4_result_id       = to_string(line[251:267])  #前走4競争成績キー
        c.pre5_result_id       = to_string(line[267:283])  #前走5競争成績キー
        c.pre1_race_id         = to_string(line[283:291])  #前走1レースキー
        c.pre2_race_id         = to_string(line[291:299])  #前走2レースキー
        c.pre3_race_id         = to_string(line[299:307])  #前走3レースキー
        c.pre4_race_id         = to_string(line[307:315])  #前走4レースキー
        c.pre5_race_id         = to_string(line[315:323])  #前走5レースキー
        c.frame_number         = to_integer(line[323])     #枠番
        c.yobi_5               = to_integer(line[324:326]) #予備
        c.show()
        return c
        
 
class ResultDatabase(BaseORM):
    """
    provide parser of raw results files and accessor for the database
    """
    def __init__(self,con):
        super(ResultDatabase,self).__init__(con,"result")

    def parse_line(self,line):
        c = Container()
        c.race_id              = to_string(line[0:8])    #レースキー
        c.horse_number         = to_integer(line[8:10])  #馬番
        c.pedigree_id          = to_string(line[10:18])  #血統登録番号
        c.registered_date      = to_string(line[18:26])  #登録日
        c.horse_name           = to_unicode(line[26:62]) #名前
        c.distance             = to_integer(line[62:66]) #距離
        c.discipline           = to_integer(line[66])    #芝ダ障害コード
        c.left_or_right        = to_integer(line[67])    #右左
        c.in_or_out            = to_integer(line[68])    #内外
        c.field_status         = to_integer(line[69:71]) #馬場状態

        c.race_category        = to_integer(line[71:73])   #種別
        c.race_condition       = to_string(line[73:75])    #条件
        c.race_remarks         = to_integer(line[75:78])   #記号
        c.race_weights         = to_integer(line[78])      #重量
        c.race_grade           = to_integer(line[79])      #グレード
        c.race_name            = to_unicode(line[80:130])  #レース名
        c.race_headcount       = to_integer(line[130:132]) #頭数
        c.race_alias           = to_unicode(line[132:140]) #レース名略称

        c.order_of_finish      = to_integer(line[140:142]) #着順
        c.irregular_category   = to_integer(line[142])     #異常区分
        c.finishing_time       = to_float(line[143:147])   #タイム
        c.basis_weight         = to_float(line[147:150])   #斤量
        c.jockey_name          = to_unicode(line[150:162]) #騎手名
        c.trainer_name         = to_unicode(line[162:174]) #調教師名
        c.odds                 = to_float(line[174:180])   #確定オッズ
        c.popularity           = to_integer(line[180:182]) #人気順位

        c.jrdb_idm             = to_float(line[182:185],0.0)   #IDM
        c.jrdb_raw_score       = to_float(line[185:188],0.0)   #素点
        c.jrdb_field_info      = to_float(line[188:191],0.0)   #馬場差
        c.jrdb_pace_info       = to_float(line[191:194],0.0)   #ペース
        c.late_start           = to_float(line[194:197],0.0)   #出遅れ
        c.position             = to_float(line[197:200],0.0)   #位置取り
        c.disadvantage         = to_float(line[200:203],0.0)   #不利
        c.disadvantage_opening = to_float(line[203:206],0.0)   #前不利
        c.disadvantage_middle  = to_float(line[206:209],0.0)   #中不利
        c.disadvantage_final   = to_float(line[209:212],0.0)   #後不利
        c.race_info            = to_float(line[212:215],0.0)   #レース
        c.course_position      = to_integer(line[215])     #コース取り
        c.condition_code       = to_integer(line[216])     #上昇度コード
        c.class_code           = to_integer(line[217:219]) #クラスコード
        c.body_code            = to_integer(line[219])     #馬体コード
        c.atmosphere_code      = to_integer(line[220])     #気配コード
        c.race_pace            = to_string(line[221])      #レースペース
        c.horse_pace           = to_string(line[222])      #馬ペース
        c.firstphase_score     = to_float(line[223:228])   #テン指数
        c.lastphase_score      = to_float(line[228:233])   #上がり指数
        c.pase_score           = to_float(line[233:238])   #ペース指数
        c.race_pace_score      = to_float(line[238:243])   #レースＰ指数
        c.first_horse_name     = to_unicode(line[243:255]) #一着馬名
        c.time_delta           = to_float(line[255:258])   #一着とのタイム差
        c.first_3f_time        = to_float(line[258:261])   #前3Fタイム
        c.last_3f_time         = to_float(line[261:264])   #後3Fタイム
        c.jrdb_remarks         = to_unicode(line[264:288]) #備考(地方競馬場名等)

        c.place_odds           = to_float(line[290:296])   #複勝確定オッズ
        c.morning_odds         = to_float(line[296:302])   #朝の時点での単勝オッズ
        c.morning_place_odds   = to_float(line[302:308])   #朝の時点での複勝オッズ
        c.pass_1               = to_integer(line[308:310]) #コーナー順位1
        c.pass_2               = to_integer(line[310:312]) #コーナー順位2
        c.pass_3               = to_integer(line[312:314]) #コーナー順位3
        c.pass_4               = to_integer(line[314:316]) #コーナー順位4
        c.first_3f_delta       = to_float(line[316:319])   #前3F地点での先頭とのタイム差
        c.last_3f_delta        = to_float(line[319:322])   #後3F地点での先頭とのタイム差
        c.jockey_id            = to_string(line[322:327])  #騎手コード
        c.trainer_id           = to_string(line[327:332])  #調教師コード
        c.weight               = to_integer(line[332:335]) #馬重量
        c.weight_delta         = to_integer(line[335:338]) #馬体重増減
        c.weather_code         = to_integer(line[338:339]) #天候コード
        c.course_info          = to_integer(line[339])     #コース
        c.race_running_style   = to_integer(line[340])     #レース脚質

        c.payback_win          = to_float(line[341:348])   #単勝払い戻し
        c.payback_place        = to_float(line[348:355])   #複勝払い戻し
        c.prize                = to_float(line[355:360])   #本賞金
        c.class_prize          = to_float(line[360:365])   #収得賞金
        c.position_at_corner   = to_integer(line[369])     #4角コース取り
        return c


class Container():
    def __getattr__(self,key):
        return self.__dict__[key]

    def __setattr__(self,key,value):
        self.__dict__[key] = value

    def __repr__(self):
        text = "Container : {0} values".format(len(self.__dict__))
        return text

    def show(self):
        text = u",\n".join([u"{0:<20}:{1}".format(k,v.value) for k,v in self.__dict__.items()])
        print(text)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self,key):
        return self.__dict__[key]
        

if __name__=="__main__":
    pass

