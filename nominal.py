#-*- coding:utf-8 -*-

def nominal_int(x,remarks):
    try:
        ret = int(x)
        return (ret,remarks)
    except:
        return (0,remarks)

def nominal_pace(x,remarks):
    if x == "H":
        return (1,3)
    if x == "M":
        return (2,3)
    elif x == "S":
        return (3,3)
    else:
        return (0,3)

FIELD_STATUS_DICT = {
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
    "42":12
}
def nominal_field_status(x,remarks):
    n = len(FIELD_STATUS_DICT)
    try:
        return (FIELD_STATUS_DICT[x],n)
    except KeyError:
        return (0,n)

PADDOK_DICT = {
    "A":1,
    "B":2,
    "C":3,
    "D":4,
    "E":5
}
def nominal_paddok(x,remarks):
    n = len(PADDOK_DICT)
    try:
        return (PADDOK_DICT[x],n)
    except KeyError:
        return (0,n)

CONDITION_SCORE_DICT = {
    "AA":1,
    "A":2,
    "B":3,
    "C":4
}
def nominal_condiction_score(x,remarks):
    n = len(CONDITION_SCORE_DICT)
    try:
        return (CONDITION_SCORE_DICT[x],n)
    except KeyError:
        return (0,n)

COURSE_INFO_DICT = {
    "AA":1,
    "A":2,
    "B":3,
    "C":4
}
def nominal_course_info(x,remarks):
    n = len(COURSE_INFO_DICT)
    try:
        return (COURSE_INFO_DICT[x],n)
    except KeyError:
        return (0,n)

CLASS_LIST = [
    "01", #芝Ｇ１
    "02", #芝Ｇ２
    "03", #芝Ｇ３
    "04", #芝ＯＰ A
    "05", #芝ＯＰ B
    "06", #芝ＯＰ C
    "07", #芝1600万A
    "08", #芝1600万B
    "09", #芝1600万C
    "10", #芝1000万A
    "11", #芝1000万B
    "12", #芝1000万C
    "13", #芝500万A
    "14", #芝500万B
    "15", #芝500万C
    "16", #芝未 A
    "17", #芝未 B
    "18", #芝未 C
    "21", #ダＧ１
    "22", #ダＧ２
    "23", #ダＧ３
    "24", #ダＯＰ Ａ
    "25", #ダＯＰ Ｂ
    "26", #ダＯＰ Ｃ
    "27", #ダ1600万Ａ
    "28", #ダ1600万Ｂ
    "29", #ダ1600万Ｃ
    "30", #ダ1000万Ａ
    "31", #ダ1000万Ｂ
    "32", #ダ1000万Ｃ
    "33", #ダ500万Ａ
    "34", #ダ500万Ｂ
    "35", #ダ500万Ｃ
    "36", #ダ未 Ａ
    "37", #ダ未 Ｂ
    "38", #ダ未 Ｃ
    "51", #障Ｇ１
    "52", #障Ｇ２
    "53", #障Ｇ３
    "54", #障ＯＰ Ａ
    "55", #障ＯＰ Ｂ
    "56", #障ＯＰ Ｃ
    "57", #障500万Ａ
    "58", #障500万Ｂ
    "59", #障500万Ｃ
    "60", #障未 Ａ
    "61", #障未 Ｂ
    "62", #障未 Ｃ
]
CLASS_DICT = {}
for i,k in enumerate(CLASS_LIST):
    CLASS_DICT[k] = i+1
def nominal_class_code(x,remarks):
    n = len(CLASS_DICT)
    try:
        return (CLASS_DICT[x],n)
    except KeyError:
        return (0,n)

RACE_COURSE_LIST = [
    "01", #札幌
    "02", #函館
    "03", #福島
    "04", #新潟
    "05", #東京
    "06", #中山
    "07", #中京
    "08", #京都
    "09", #阪神
    "10", #小倉
    "21", #旭川
    "22", #札幌
    "23", #門別
    "24", #函館
    "25", #盛岡
    "26", #水沢
    "27", #上山
    "28", #新潟
    "29", #三条
    "30", #足利
    "31", #宇都
    "32", #高崎
    "33", #浦和
    "34", #船橋
    "35", #大井
    "36", #川崎
    "37", #金沢
    "38", #笠松
    "39", #名古
    "40", #中京
    "41", #園田
    "42", #姫路
    "43", #益田
    "44", #福山
    "45", #高知
    "46", #佐賀
    "47", #荒尾
    "48", #中津
    "61", #英国
    "62", #愛国
    "63", #仏国
    "64", #伊国
    "65", #独国
    "66", #米国
    "67", #加国
    "68", #UAE
    "69", #豪州
    "70", #新国
    "71", #香港
    "72", #チリ
    "73", #星国
    "74", #瑞国
    "75", #マカ
    "76" #墺国
]
RACE_COURSE_DICT = {}
for i,k in enumerate(RACE_COURSE_LIST):
    RACE_COURSE_DICT[k] = i+1
def nominal_race_course_code(x,remarks):
    n = len(RACE_COURSE_LIST)
    try:
        return(RACE_COURSE_DICT[x],n)
    except KeyError:
        return(0,n)

JRA_COURSE_LIST = [
    "01", #札幌
    "02", #函館
    "03", #福島
    "04", #新潟
    "05", #東京
    "06", #中山
    "07", #中京
    "08", #京都
    "09", #阪神
    "10" #小倉
]
JRA_COURSE_DICT = {}
for i,k in enumerate(JRA_COURSE_LIST):
    JRA_COURSE_DICT[k] = i+1
def nominal_jra_course_code(x,remarks):
    n = len(JRA_COURSE_DICT)
    try:
        return (JRA_COURSE_DICT[x],n)
    except KeyError:
        return (0,n)

RACE_KIND_LIST = [
    "11",
    "12",
    "13",
    "14",
    "20",
]
RACE_KIND_DICT = {}
for i,k in enumerate(RACE_KIND_LIST):
    RACE_KIND_DICT[k] = i+1
def nominal_race_kind(x,remarks):
    n = len(RACE_KIND_LIST)
    try:
        return (RACE_KIND_DICT[x],n)
    except KeyError:
        return (0,n)

RACE_REQUIREMENTS_LIST = [
    "04",
    "05",
    "08",
    "09",
    "10",
    "15",
    "16",
    "A1",
    "A2",
    "A3",
    "OP",
]
RACE_REQUIREMENTS_DICT = {}
for i,k in enumerate(RACE_REQUIREMENTS_LIST):
    RACE_REQUIREMENTS_DICT[k] = i+1
def nominal_race_requirements(x,remarks):
    n = len(RACE_REQUIREMENTS_LIST)
    try:
        return (RACE_REQUIREMENTS_DICT[x],n)
    except KeyError:
        return (0,n)

PADDOCK_RANK_LIST = [
    "A",
    "B",
    "C",
    "D",
    "E",
]
PADDOCK_RANK_DICT = {}
for i,k in enumerate(PADDOCK_RANK_LIST):
    PADDOCK_RANK_DICT[k] = i+1
def nominal_paddock_rank(x,remarks):
    n = len(PADDOCK_RANK_DICT)
    try:
        return (PADDOCK_RANK_DICT[x],n)
    except KeyError:
        return (0,n)

PEDIGREE_TYPE_LS = [
    "1101","1102","1103","1104","1105","1106","1107","1108","1109",
    "1201","1202","1203","1204","1205","1206","1207",
    "1301","1302","1303","1304","1305","1306","1307","1308","1309","1310","1311","1312",
    "1401","1402","1403",
    "1501","1502","1503",
    "1601","1602","1603",
    "1701","1702",
    "1801","1802","1803","1804","1805","1806","1807",
    "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912","1913",
    "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
    "2101","2102","2103","2104","2105",
    "2201","2202","2203","2204","2205","2206",
    "2301",
    "2401",
]
PEDIGREE_TYPE_DICT = {}
for i,k in enumerate(PEDIGREE_TYPE_LS):
    PEDIGREE_TYPE_DICT[k] = i+1
def nominal_pedigree_type(x,remarks):
    n = len(PEDIGREE_TYPE_DICT)
    try:
        return (PEDIGREE_TYPE_DICT[x],n)
    except KeyError:
        return (0,n)

if __name__=="__main__":
    a = nominal_jra_course_code("01",None)
    print(a)
