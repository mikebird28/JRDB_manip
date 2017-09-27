#-*- coding:utf-8 -*-

from util import *
import nominal

#functions which convert data to appropriate type
def to_integer(x,illegal_value = None):
    """
    convert x to integer
    if x is not integer, return None

    Args:
        x : variable which you want to convert to integer 
        illegal_value : value which use when x is not convetable

    """
    #x = x.strip()
    try:
        return Maybe(INT_SYNBOL,int(x))
    except ValueError:
        return Maybe(INT_SYNBOL,illegal_value)

def to_float(x,illegal_value = None):
    try:
        return Maybe(FLO_SYNBOL,float(x))
    except ValueError:
        return Maybe(FLO_SYNBOL,illegal_value)

def to_string(x,illegal_value = None):
    text = x.strip()
    if text == "":
        return Maybe(STR_SYNBOL,None)
    else:
        return Maybe(STR_SYNBOL,text)

def to_unicode(x,illegal_value = None):
    text = x.decode("cp932").strip()
    if text == u"":
        return Maybe(UNI_SYNBOL,None)
    else:
        return Maybe(UNI_SYNBOL,text)

def to_nominal(x,converter = nominal.nominal_int, n = 1):
    x = x.strip()
    value = converter(x)
    if value > x:
        raise Exception("nominal convert error")
    return Maybe("NOM",(value,n))

class ColumnInfo(object):
    def __init__(self,column_name,table_name,typ,n = None):
        self.column_name = column_name
        self.table_name = table_name
        self.typ = typ
        self.n = n

    def __repr__(self):
        return "{0} {1} {2}".format(self.column_name,self.table_name,self.typ)

class ColumnInfoORM(object):

    def __init__(self,con):
        self.con = con
        if not table_exists(con,"column_info"):
            self.__create_table()

    def __create_table(self):
        sql = """CREATE TABLE column_info (
                 id INTEGER primary key,
                 column_name TEXT,
                 table_name TEXT,
                 type TEXT,
                 n INTEGER
                 );"""
        self.con.execute(sql)
        self.con.commit()

    def insert(self,column_name,table_name,typ,n=None):
        keys_text = "column_name,table_name,type,n"
        sql = u"insert into {0}({1}) values(?,?,?,?);".format("column_info",keys_text)
        self.con.execute(sql,[column_name,table_name,typ,n])
        self.con.commit()

    def column_dict(self,table_name):
        dic = {}
        sql = "SELECT column_name,table_name,type,n FROM column_info WHERE table_name = '{0}'".format(table_name)
        cur = self.con.execute(sql)
        for row in cur:
            column_name = row[0]
            table_name = row[1]
            typ = row[2]
            n = row[3]
            ci = ColumnInfo(column_name,table_name,typ,n)
            dic[column_name] = ci
        return dic

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

def create_table(con,name,type_dict,nominal_dict,unique_ls = []):
    """create new tables in the given connectio
    Args:
        con       : database connection
        name      : table name
        type_dict : dictionary of columns and type
    Returns:
        return True if transaction was successed
        return False if transaction was failed
    """
    ci_orm = ColumnInfoORM(con)
    type_to_str = {STR_SYNBOL:"TEXT",UNI_SYNBOL:"TEXT",INT_SYNBOL:"INTEGER",FLO_SYNBOL:"REAL",NOM_SYNBOL:"INTEGER"}
    type_tuples = []
    keys = type_dict.keys()
    keys.sort()
    for k in keys:
        ci_orm.insert(k,name,type_dict[k],nominal_dict[k])
        ctype = type_dict[k]
        type_str = type_to_str[ctype]
        is_unique = " primary key" if k in unique_ls else ""
        type_tuples.append((k,type_str,is_unique))
    columns = ",".join(["{0} {1}{2}".format(k,v,u) for k,v,u in type_tuples])
    sql = """CREATE TABLE {name}({columns});""".format(name = name,columns = columns)
    con.execute(sql)

def insert_container(con,name,container):
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
    for k,v in container.items():
        if v.value == None:
            continue
        keys.append(k)
        values.append(v.string())

    keys_text = u",".join(keys)
    values_text = u",".join(values)
    try:
        sql = u"insert into {0}({1}) values({2});".format(name,keys_text,values_text)
        con.execute(sql)
    except Exception as e:
        print(e)
        print("")

def set_index(con,index_name,table_name,columns):
    columns_txt = ",".join(columns)
    sql = "CREATE INDEX {0} ON {1}({2});".format(index_name,table_name,columns_txt)
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
            type_dict = {k:bef_c[k].typ for k in bef_c.keys()}
            continue
        for k in c.keys():
            if c[k].typ != bef_c[k].typ:
                raise Exception("Container attribute's type is something wrong")
    return type_dict

def inference_nominal(containers):
    nominal_dict = {}
    is_fisrt = True
    bef_c = None
    for col in containers[0].keys():
        if containers[0][col].typ == NOM_SYNBOL:
            nominal_dict[col] = containers[0][col].value[1]
        else:
            nominal_dict[col] = None
    return nominal_dict

class BaseORM(object):
    def __init__(self,con,table_name):
        self.table_name = table_name
        self.con = con
        self.table_exists = table_exists(self.con,self.table_name)

    def insert_file(self,fp):
        containers = []
        for line in fp.readlines():
            c = self.parse_line(line)
            containers.append(c)

        if not self.table_exists:
            type_dict = inference_type(containers)
            nom_dict = inference_nominal(containers)
            nominal_dict = inference_nominal(containers)
            unique_ls = self.set_unique()
            create_table(self.con,self.table_name,type_dict,nominal_dict,unique_ls)
            self.set_indexes()
            self.table_exists = True

        for c in containers:
            if self.check_container(c):
                insert_container(self.con,self.table_name,c)
        self.con.commit()

    def check_container(self,c):
        return True

    def set_unique(self):
        return []
 

class PayoffDatabase(BaseORM):
    def __init__(self,con):
        super(PayoffDatabase,self).__init__(con,"payoff")


    def parse_line(self,line):
        c = Container()
        c.race_id = to_string(line[0:8])

        c.win_horse_1  = to_integer(line[8:10])
        c.win_payoff_1 = to_integer(line[10:17])
        c.win_horse_2  = to_integer(line[17:19])
        c.win_payoff_2 = to_integer(line[19:26])
        c.win_horse_3  = to_integer(line[26:28])
        c.win_payoff_3 = to_integer(line[28:35])

        c.place_horse_1   = to_integer(line[35:37])
        c.place_payoff_1  = to_integer(line[37:44])
        c.place_horse_2   = to_integer(line[44:46])
        c.place_payoff_2  = to_integer(line[46:53])
        c.place_horse_3   = to_integer(line[53:55])
        c.place_payoff_3  = to_integer(line[62:69])
        c.place_horse_4   = to_integer(line[69:71])
        c.place_payoff_4  = to_integer(line[71:78])
        c.place_horse_5   = to_integer(line[78:80])
        c.place_payoff_5  = to_integer(line[80:87])
        return c

    def set_indexes(self):
        set_index(self.con,"pd_race_id_idx",self.table_name,["race_id"])

    def set_unique(self):
        return ["race_id"]

class HorseInfoDatabase(BaseORM):
    def __init__(self,con):
        super(HorseInfoDatabase,self).__init__(con,"horse_info")

    def parse_line(self,line):
        c = Container()
        c.race_id              = to_string(line[0:8])    #レースキー
        c.horse_number         = to_integer(line[8:10])  #馬番
        c.horse_id             = to_string(line[0:10])  #馬キー
        c.pedigree_id          = to_integer(line[10:18])  #血統登録番号
        #c.horse_name           = to_unicode(line[18:54]) #名前

        c.idm                  = to_float(line[54:59])   #IDM
        c.jockey_score         = to_float(line[59:64])   #騎手指数
        c.info_score           = to_float(line[64:69])   #情報指数
        c.composite_score      = to_float(line[84:89])   #総合指数

        c.running_style        = to_integer(line[89])    #脚質
        c.distance_fitness     = to_nominal(line[90],n=6)#距離適性
        c.condiction_score     = to_integer(line[91])    #上昇度
        c.rotation             = to_integer(line[92:95]) #ローテーション

        c.base_odds            = to_float(line[95:100])  #基準オッズ
        c.base_popularity      = to_integer(line[100:102]) #基準人気順位
        c.base_place_odds      = to_float(line[102:107])   #基準複勝オッズ
        c.base_place_popularity = to_integer(line[107:109])#基準複勝人気順位
        c.specific_info_5      = to_integer(line[109:112],0) #特定情報◎
        c.specific_info_4      = to_integer(line[112:115],0) #特定情報○
        c.specific_info_3      = to_integer(line[115:118],0) #特定情報▲
        c.specific_info_2      = to_integer(line[118:121],0) #特定情報△
        c.specific_info_1      = to_integer(line[121:124],0) #特定情報×
        c.general_info_5       = to_integer(line[124:127],0) #総合情報◎
        c.general_info_4       = to_integer(line[127:130],0) #総合指数○
        c.general_info_3       = to_integer(line[130:133],0) #総合指数▲
        c.general_info_2       = to_integer(line[133:136],0) #総合指数△
        c.general_info_1       = to_integer(line[136:139],0) #総合指数×
        c.popurality_score     = to_integer(line[139:144]) #人気指数
        c.training_score       = to_float(line[144:149]) #調教師数
        c.stable_score         = to_float(line[149:154]) #厩舎指数

        c.training_sign_code   = to_integer(line[154])     #調教矢印コード
        c.stable_eval_code     = to_integer(line[155])     #厩舎評価コード
        c.jockey_quinella      = to_float(line[156:160])   #騎手期待値連対率
        c.running_score        = to_integer(line[160:163]) #激走指数
        c.hoof_code            = to_integer(line[163:165]) #蹄コード
        c.heavy_fitness_code   = to_integer(line[165])     #重適正コード
        c.class_code           = to_integer(line[166:168]) #クラスコード

        c.brinker              = to_nominal(line[170],n=3)     #ブリンカー
        #c.jockey_name          = to_unicode(line[171:183]) #騎手名
        c.basis_weight         = to_integer(line[183:186]) #負担重量
        c.apprentice_class     = to_nominal(line[186],n=3) #見習い区分
        #c.trainer_name         = to_unicode(line[187:199]) #調教師名
        #c.trainer_division     = to_unicode(line[199:203]) #調教師所属

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

        c.composite_sign       = to_nominal(line[326],n=6)     #総合印
        c.idm_sign             = to_nominal(line[327],n=6)     #IDM印
        c.info_sign            = to_nominal(line[328],n=6)     #情報印
        c.jockey_sign          = to_nominal(line[329],n=6)     #騎手印
        c.stable_sign          = to_nominal(line[330],n=6)     #厩舎印
        c.training_sign        = to_nominal(line[331],n=6)     #調教印
        c.hard_running_sign    = to_nominal(line[332],n=6)     #激走印
        c.turf_fitness         = to_nominal(line[333],n=3)     #芝適正コード
        c.dirt_fitness         = to_nominal(line[334],n=3)     #ダ適正コード
        c.jockey_code          = to_integer(line[335:340]) #騎手コード
        c.trainer_code         = to_integer(line[340:345]) #調教師コード

        c.prize                = to_integer(line[346:352]) #獲得賞金
        c.class_prize          = to_integer(line[352:357]) #収得賞金
        c.condition_class      = to_integer(line[357])     #条件クラス

        c.firstphase_score     = to_float(line[358:363])   #テン指数
        c.pace_score           = to_float(line[363:368])   #ペース指数
        c.lastphase_score      = to_float(line[368:373])   #上がり指数
        c.position_score       = to_float(line[373:378])   #位置指数
        c.pace_prediction      = to_nominal(line[378], n=3, converter=nominal.nominal_pace)  #ペース予想
        c.middlephase_order    = to_integer(line[379:381]) #道中順位
        c.middlephase_delta    = to_integer(line[381:383]) #道中差
        c.middlephase_in_out   = to_integer(line[383])     #道中内外
        c.lastphase_order      = to_integer(line[384:386]) #後3F順位
        c.lastphase_delta      = to_integer(line[386:388]) #後3F差
        c.lastphase_in_out     = to_integer(line[388])     #後3F内外
        c.oof_prediction       = to_integer(line[389:391]) #ゴール順位
        c.oof_delta            = to_integer(line[391:393]) #ゴール差
        c.oof_in_out           = to_integer(line[393])     #ゴール内外
        c.development_sign     = to_integer(line[394])     #展開記号

        c.distance_fitness2    = to_nominal(line[395],n=6)     #距離適性2
        c.weight_afd           = to_integer(line[396:399]) #枠確定後馬体重
        c.weight_delta_afd     = to_integer(line[399:402]) #枠確定馬体重増減

        c.cancel_flag          = to_nominal(line[402],n=1) #取り消しフラグ
        c.sex_code             = to_integer(line[403])     #性別コード
        #c.owner_name           = to_unicode(line[404:444]) #馬主名
        c.owner_code           = to_integer(line[444:446]) #馬主会コード

        c.horse_sign_code        = to_integer(line[446:448]) #馬記号コード
        c.hard_runnning_order    = to_integer(line[448:450]) #激走順位
        c.ls_score_order         = to_integer(line[450:452]) #LS指数順位
        c.firstphase_score_order = to_integer(line[452:454]) #テン指数順位
        c.pace_score_order       = to_integer(line[454:456]) #ペース指数順位
        c.lastphase_score_order  = to_integer(line[456:458]) #上がり指数順位
        c.position_score_order   = to_integer(line[458:460]) #位置指数順位

        c.jockey_win           = to_float(line[460:464])     #騎手期待値勝率
        c.jockey_place         = to_float(line[464:468])     #騎手期待３着内率
        c.transport_class      = to_integer(line[468])       #輸送区分

        c.running_style        = to_integer(line[469:477])   #走法
        c.body_shape           = to_unicode(line[477:501])   #体型
        #c.body_composite_1     = to_integer(line[501:504])   #体型総合1
        #c.body_composite_2     = to_integer(line[504:507])   #体型総合2
        #c.body_composite_3     = to_integer(line[507:510])   #体型総合3
        c.horse_remarks_1      = to_integer(line[510:513])   #馬特記1
        c.horse_remarks_2      = to_integer(line[513:516])   #馬特記2
        c.horse_remarks_3      = to_integer(line[516:519])   #馬特記3

        c.start_score           = to_float(line[519:523],0.0)    #馬スタート指数
        c.late_start_per        = to_float(line[523:527],0.0)    #馬出遅れ率
        #c.last_run_reference    = to_integer(line[527:529])  #参考前走
        c.last_run_jockey_code  = to_integer(line[529:534])  #参考前走騎手コード
        c.precious_ticket_score = to_integer(line[534:537])  #万馬券指数
        c.precious_ticket_sign  = to_integer(line[537])      #万馬券印

        c.downgrading_flag     = to_nominal(line[538],n=2)       #降級フラグ
        #c.hard_running_type    = to_unicode(line[539:541])   #激走タイプ
        #c.recuperation_type    = to_integer(line[541:543])   #休養理由分類コード
        c.remarks_flag         = to_unicode(line[543:559])   #フラグ

        c.stabling_race_count    = to_integer(line[559:561]) #入厩何走目
        #c.stabling_date          = to_string(line[561:569])  #入厩年月日
        c.stabling_elapsed_dates = to_integer(line[569:572],0) #入厩何日目


        #c.paddock              = to_unicode(line[572:622])  #放牧先
        c.paddock_rank         = to_string(line[622])       #放牧先ランク
        c.stable_rank          = to_string(line[623])       #厩舎ランク
        return c

    def check_container(self,container):
        return True
        
    def set_indexes(self):
        set_index(self.con,"hid_race_horse_id_idx",self.table_name,["race_id","horse_id"])
        set_index(self.con,"hid_horse_id_idx",self.table_name,["horse_id"])
        set_index(self.con,"hid_race_id_idx",self.table_name,["race_id"])

    def set_unique(self):
        return ["horse_id"]



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

        c.result_id            = to_string(line[10:26])
        c.pedigree_id          = to_string(line[10:18])  #血統登録番号
        c.registered_date      = to_string(line[18:26])  #登録日

        #c.horse_name           = to_unicode(line[26:62]) #名前
        c.distance             = to_integer(line[62:66]) #距離
        c.discipline           = to_nominal(line[66],n = 3)    #芝ダ障害コード
        c.left_or_right        = to_nominal(line[67],n = 3)    #右左
        c.in_or_out            = to_nominal(line[68],n = 3)    #内外
        c.field_status         = to_integer(line[69:71]) #馬場状態

        c.race_category        = to_integer(line[71:73])   #種別
        c.race_condition       = to_string(line[73:75])    #条件
        c.race_remarks         = to_integer(line[75:78])   #記号
        c.race_weights         = to_integer(line[78])      #重量
        c.race_grade           = to_integer(line[79])      #グレード
        #c.race_name            = to_unicode(line[80:130])  #レース名
        c.race_headcount       = to_integer(line[130:132]) #頭数
        #c.race_alias           = to_unicode(line[132:140]) #レース名略称

        c.order_of_finish      = to_integer(line[140:142]) #着順
        c.irregular_category   = to_integer(line[142])     #異常区分
        c.finishing_time       = to_float(line[143:147])   #タイム
        c.basis_weight         = to_float(line[147:150])   #斤量
        #c.jockey_name          = to_unicode(line[150:162]) #騎手名
        #c.trainer_name         = to_unicode(line[162:174]) #調教師名
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
        c.pace_score           = to_float(line[233:238])   #ペース指数
        c.race_pace_score      = to_float(line[238:243])   #レースＰ指数
        #c.first_horse_name     = to_unicode(line[243:255]) #一着馬名
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

        c.payback_win          = to_float(line[341:348],0.0)   #単勝払い戻し
        c.payback_place        = to_float(line[348:355],0.0)   #複勝払い戻し
        c.prize                = to_float(line[355:360],0.0)   #本賞金
        c.class_prize          = to_float(line[360:365],0.0)   #収得賞金
        c.position_at_corner   = to_integer(line[369])     #4角コース取り
        return c

    def check_container(self,container):
        if container.irregular_category.value != 0:
            return False
        return True

    def set_indexes(self):
       set_index(self.con,"rd_result_id_idx",self.table_name,["race_id","result_id"])

    def set_unique(self):
        return ["result_id"]
 
class ExpandedInfoDatabase(BaseORM):
    def __init__(self,con):
        super(ExpandedInfoDatabase,self).__init__(con,"exinfo")

    def parse_line(self,line):
        c = Container()
        c.race_id      = to_string(line[0:8])
        c.horse_number = to_integer(line[8:10])
        c.horse_id     = to_string(line[0:10])

        c.jra_win             = to_integer(line[10:13],0)
        c.jra_second          = to_integer(line[13:16],0)
        c.jra_third           = to_integer(line[16:19],0)
        c.jra_lose            = to_integer(line[19:22],0)
        c.jra_place           = to_integer(c.jra_win.value + c.jra_second.value + c.jra_third.value)
        c.jra_total           = to_integer(c.jra_place.value + c.jra_lose.value)
        c.jra_win_per         = to_float(divide(c.jra_win.value,c.jra_total.value))
        c.jra_place_per       = to_float(divide(c.jra_place.value,c.jra_total.value))

        c.interleague_win       = to_integer(line[22:25],0)
        c.interleague_second    = to_integer(line[25:28],0)
        c.interleague_third     = to_integer(line[28:31],0)
        c.interleague_lose      = to_integer(line[31:34],0)
        c.interleague_place     = to_integer(c.interleague_win.value + c.interleague_second.value + c.interleague_third.value)
        c.interleague_total     = to_integer(c.interleague_place.value + c.interleague_lose.value)
        c.interleague_win_per   = to_float(divide(c.interleague_win.value,c.interleague_total.value))
        c.interleague_place_per = to_float(divide(c.interleague_place.value,c.interleague_total.value))

        c.others_win          = to_integer(line[34:37],0)
        c.others_second       = to_integer(line[37:40],0)
        c.others_third        = to_integer(line[40:43],0)
        c.others_lose         = to_integer(line[43:46],0)
        c.others_place        = to_integer(c.others_win.value + c.others_second.value + c.others_third.value)
        c.others_total        = to_integer(c.others_place.value + c.others_lose.value)
        c.others_win_per      = to_float(divide(c.others_win.value,c.others_total.value))
        c.others_place_per    = to_float(divide(c.others_place.value,c.others_total.value))

        c.surf_win            = to_integer(line[46:49],0)
        c.surf_second         = to_integer(line[49:52],0)
        c.surf_third          = to_integer(line[52:55],0)
        c.surf_lose           = to_integer(line[55:58],0)
        c.surf_place          = to_integer(c.surf_win.value + c.surf_second.value + c.surf_third.value)
        c.surf_total          = to_integer(c.surf_place.value + c.surf_lose.value)
        c.surf_win_per        = to_float(divide(c.surf_win.value,c.surf_total.value))
        c.surf_place_per      = to_float(divide(c.surf_place.value,c.surf_total.value))

        c.surf_dist_win       = to_integer(line[58:61],0)
        c.surf_dist_second    = to_integer(line[61:64],0)
        c.surf_dist_third     = to_integer(line[64:67],0)
        c.surf_dist_lose      = to_integer(line[67:70],0)
        c.surf_dist_place          = to_integer(c.surf_dist_win.value + c.surf_dist_second.value + c.surf_dist_third.value)
        c.surf_dist_total          = to_integer(c.surf_dist_place.value + c.surf_dist_lose.value)
        c.surf_dist_win_per        = to_float(divide(c.surf_dist_win.value,c.surf_dist_total.value))
        c.surf_dist_place_per      = to_float(divide(c.surf_dist_place.value,c.surf_dist_total.value))

        c.dist_win            = to_integer(line[70:73],0)
        c.dist_second         = to_integer(line[73:76],0)
        c.dist_third          = to_integer(line[76:79],0)
        c.dist_lose           = to_integer(line[79:82],0)
        c.dist_place          = to_integer(c.dist_win.value + c.dist_second.value + c.dist_third.value)
        c.dist_total          = to_integer(c.dist_place.value + c.dist_lose.value)
        c.dist_win_per        = to_float(divide(c.dist_win.value,c.dist_total.value))
        c.dist_place_per      = to_float(divide(c.dist_place.value,c.dist_total.value))

        c.rotation_first      = to_integer(line[82:85],0)
        c.rotation_second     = to_integer(line[85:88],0)
        c.rotation_third      = to_integer(line[88:91],0)
        c.rotation_lose       = to_integer(line[91:94],0)

        c.course_first        = to_integer(line[94:97],0)
        c.course_second       = to_integer(line[97:100],0)
        c.course_third        = to_integer(line[100:103],0)
        c.course_lose         = to_integer(line[103:106],0)

        c.jockey_first        = to_integer(line[106:109],0)
        c.jockey_second       = to_integer(line[109:112],0)
        c.jockey_third        = to_integer(line[112:115],0)
        c.jockey_lose         = to_integer(line[115:118],0)

        c.surf_good_first        = to_integer(line[118:121],0)
        c.surf_good_second       = to_integer(line[121:124],0)
        c.surf_good_third        = to_integer(line[124:127],0)
        c.surf_good_lose         = to_integer(line[127:130],0)

        c.surf_middle_first        = to_integer(line[130:133],0)
        c.surf_middle_second       = to_integer(line[133:136],0)
        c.surf_middle_third        = to_integer(line[136:139],0)
        c.surf_middle_lose         = to_integer(line[139:142],0)

        c.surf_bad_first        = to_integer(line[142:145],0)
        c.surf_bad_second       = to_integer(line[145:148],0)
        c.surf_bad_third        = to_integer(line[148:151],0)
        c.surf_bad_lose         = to_integer(line[151:154],0)

        c.pace_slow_first        = to_integer(line[142:145],0)
        c.pace_slow_second       = to_integer(line[145:148],0)
        c.pace_slow_third        = to_integer(line[148:151],0)
        c.pace_slow_lose         = to_integer(line[151:154],0)

        c.pace_middle_first        = to_integer(line[154:157],0)
        c.pace_middle_second       = to_integer(line[157:160],0)
        c.pace_middle_third        = to_integer(line[160:163],0)
        c.pace_middle_lose         = to_integer(line[163:166],0)

        c.pace_high_first        = to_integer(line[166:169],0)
        c.pace_high_second       = to_integer(line[169:172],0)
        c.pace_high_third        = to_integer(line[172:175],0)
        c.pace_high_lose         = to_integer(line[175:178],0)

        c.season_win          = to_integer(line[178:181],0)
        c.season_second       = to_integer(line[181:184],0)
        c.season_third        = to_integer(line[184:187],0)
        c.season_lose         = to_integer(line[187:190],0)
        c.season_place          = to_integer(c.season_win.value + c.season_second.value + c.season_third.value)
        c.season_total          = to_integer(c.season_place.value + c.season_lose.value)
        c.season_win_per        = to_float(divide(c.season_win.value,c.season_total.value))
        c.season_place_per      = to_float(divide(c.season_place.value,c.season_total.value))

        c.frame_first        = to_integer(line[190:193],0)
        c.frame_second       = to_integer(line[193:196],0)
        c.frame_third        = to_integer(line[196:199],0)
        c.frame_lose         = to_integer(line[199:202],0)
        return c

    def set_indexes(self):
        #set_index(self.con,"ex_race_id_idx",self.table_name,["race_id"])
        set_index(self.con,"ex_horse_id_idx",self.table_name,["horse_id"])

    def check_container(self,container):
        return True

    def set_unique(self):
        return ["horse_id"]
 

def create_feature_table(con):
    raw_columns,columns_dict,columns_query = fetch_columns_info(con)

    fixed_columns = raw_columns
    fixed_columns.append("is_win")
    fixed_columns.append("is_place_win")
    columns_dict["is_win"] = ColumnInfo("is_win","feature",INT_SYNBOL)
    columns_dict["is_place_win"] = ColumnInfo("is_place_win","feature",INT_SYNBOL)
    columns_txt = ",".join(columns_query)

    # check if feature table exist
    if not table_exists(con,"feature"):
        sql_create = "CREATE TABLE feature({0})".format(",".join(fixed_columns))
        con.execute(sql_create)
        con.commit()
        ci_orm = ColumnInfoORM(con)
        for v in columns_dict.values():
            ci_orm.insert(v.column_name,v.table_name,v.typ,v.n)
    columns = fixed_columns

    # add the optional column
    sql = """SELECT {0} FROM horse_info as hi
             LEFT JOIN payoff on hi.race_id = payoff.race_id
             LEFT JOIN exinfo on hi.horse_id = exinfo.horse_id
             LEFT JOIN result as p1 ON hi.pre1_result_id = p1.result_id
             LEFT JOIN result as p2 ON hi.pre2_result_id = p2.result_id
             LEFT JOIN result as p3 ON hi.pre3_result_id = p3.result_id
             LEFT JOIN result as p4 ON hi.pre4_result_id = p4.result_id
             LEFT JOIN result as p5 ON hi.pre5_result_id = p5.result_id;""".format(columns_txt)

    cur = con.execute(sql)
    for count,row in enumerate(cur):
        if count % 100 == 0:
            print(count)
            con.commit()
        container = Container()
        for col_name,value in zip(columns,row):
            typ = columns_dict[col_name].typ
            maybe = Maybe(typ,value)
            container[col_name] = maybe
        container["is_win"] = to_integer(container.payoff_win_horse_1.value == container.info_horse_number.value)
        is_place_1 = container.payoff_place_horse_1.value == container.info_horse_number.value
        is_place_2 = container.payoff_place_horse_2.value == container.info_horse_number.value
        is_place_3 = container.payoff_place_horse_3.value == container.info_horse_number.value
        container["is_place_win"] = to_integer(is_place_1 or is_place_2 or is_place_3)
        insert_container(con,"feature",container)
    con.commit()


def fetch_columns_info(con):
    columns_list = []
    columns_dict = {}
    columns_query = []
    ci_orm = ColumnInfoORM(con)

    #create target columns list and dictionary
    hi_raw_col = column_list(con,"horse_info")
    for c in hi_raw_col:
        columns_list.append("info_{0}".format(c))
        columns_query.append("hi.{0} as 'info_{0}'".format(c))
    hi_fixed = fixed_column_dict(con,"horse_info","info")
    columns_dict.update(hi_fixed)

    po_raw_col = column_list(con,"payoff")
    for c in po_raw_col:
        columns_list.append("payoff_{0}".format(c))
        columns_query.append("payoff.{0} as 'payoff_{0}'".format(c))
    po_fixed = fixed_column_dict(con,"payoff","payoff")
    columns_dict.update(po_fixed)

    res_raw_col = column_list(con,"result")
    for i in range(5):
        for c in res_raw_col:
            columns_list.append("pre{0}_{1}".format(i+1,c))
            columns_query.append("p{0}.{1} as 'pre{0}_{1}'".format(i+1,c))
        res_fixed = fixed_column_dict(con,"result","pre{0}".format(i+1))
        columns_dict.update(res_fixed)

    ex_raw_col = column_list(con,"exinfo")
    for c in ex_raw_col:
        columns_list.append("exinfo_{0}".format(c))
        columns_query.append("exinfo.{0} as 'exinfo_{0}'".format(c))
    ex_fixed = fixed_column_dict(con,"exinfo","exinfo")
    columns_dict.update(ex_fixed)

    return (columns_list,columns_dict,columns_query)


def fixed_column_dict(con,table_name,prefix):
    ci_orm = ColumnInfoORM(con)
    fixed_columns_dict = {}
    raw_columns_dict = ci_orm.column_dict(table_name)
    for col in raw_columns_dict.keys():
        cn = "{0}_{1}".format(prefix,col)
        tn = "feature"
        typ = raw_columns_dict[col].typ
        n = raw_columns_dict[col].n
        ci = ColumnInfo(cn,tn,typ,n)
        fixed_columns_dict[cn] = ci
    return fixed_columns_dict


def column_list(con,table_name):
    columns = []
    sql = "SELECT * FROM {0}".format(table_name)
    cur = con.execute(sql)
    for column in cur.description:
        name = column[0]
        columns.append(name)
    return columns

def divide(a,b):
    try:
        return float(a)/b
    except ZeroDivisionError:
        return 0.0
