try:
    import sqlqueryV2 as sqq
except:
    pass

try:
    import RAW.int
except:
    pass

import warnings
warnings.filterwarnings("ignore")
import gc
import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.set_option("display.max_row", 500)
from importlib import reload

def add_guests(file_name):
    os.system("""echo "{0}%%%{1}">>/home/bozb/lsy/CACHE/{2}""". \
              format(os.getcwd(), datetime.now(), file_name))
add_guests("default_user_cnt.txt")

@property
def h1(self):
    return self.head(1).T
pd.DataFrame.h1 = h1

def r1(self, s):
    self.columns = pd.Series(self.columns).apply(lambda x:s.get(x, x))
pd.DataFrame.r1 = r1

def cc(self, s):
    return pd.Series(self.columns[self.columns.str.contains(s)])
pd.DataFrame.cc = cc

def ncc(self, s):
    return pd.Series(self.columns[~self.columns.str.contains(s)])
pd.DataFrame.ncc = ncc


def cc_s(self, s):
    return self[self.str.contains(s)]

def ncc_s(self, s):
    return self[~self.str.contains(s)]

def cc_si(self, s):
    return self[self.index.str.contains(s)]

def cv(self, s):
    return pd.Series({i: self.get(i, i) for i in s}, name = self.name)

pd.Series.cv = cv
pd.Series.cc = cc_s
pd.Series.ncc = ncc_s
pd.Series.cci = cc_si

## bf1 bf -- binning function -----------------------------

def bf1(word):
    return lambda x:{"cnt": x.shape[0], "rate": x[word]. mean()}

bf = lambda y:{"cnt": y.shape[0], "mean": y. mean()}

def unstack(_df2, ind):
    _df2_rate = _df2.set_index(["code_0", "code_1", "bin_0", "bin_1"])[ind]. unstack(["code_0", "bin_0"])
    _df2_rate.index = _df2_rate.index.get_level_values(1)
    _df2_rate.columns = _df2_rate.columns.get_level_values(1)
    _df2_rate.index.name = ind
    _df2_rate.columns.name = ind
    return _df2_rate

##*****************************************************************************************

HOME_DIR = "/home/bozb/lsy/"
RAW_DATA_DIR = "/home/bozb/lsy/DATA/Y_CACHE/"
VRB_DATA_DIR = "/home/bozb/lsy/DATA/X_CACHE/"
CMT_DATA_DIR = "/home/bozb/lsy/DATA/CMT/"
MODEL_DATA_DIR = "/home/bozb/lsy/DATA/MODEL/"

def read_raw(path):
    df = pd.read_pickle(RAW_DATA_DIR + path)
    return df

def read_vrb(path):
    df = pd.read_pickle(VRB_DATA_DIR + path)
    return df

def read_cmt(*l, **kwargs):
    cmt = pd.read_pickle(CMT_DATA_DIR + "cmt_tianyan.pkl")
    cmt1 = pd.read_pickle(CMT_DATA_DIR + "cmt_zbank.pkl")
    cmt1 = cmt1[(~cmt1.isnull()) & (~cmt1.index.isnull())]
    cmt1_1 = cmt1.copy()
    cmt1.index = cmt1.index.str.lower()
    cmt2 = pd.read_pickle(CMT_DATA_DIR + "cmt_td.pkl")
    cmt = pd.concat([cmt, cmt1, cmt1_1, cmt2]).reset_index().drop_duplicates("index").set_index("index").iloc[:, 0]
    return cmt

class file_dt:
    def __init__(self, path, dt = None):
        if dt is None:
            dt = datetime.now() - timedelta(1)
        self.dt1 = pd.to_datetime(dt)
        self.dt = self.dt1.strftime("%Y%m%d")
        self.path = path + ".pkl"
        self.path_dt = path + "_dt.pkl"
    def hasdt(self):
        if os.path.exists(self.path_dt) and os.path.exists(self.path):
            return True
        else:
            return False
    def read(self):
        _tmp = pd.read_pickle(self.path)
        return _tmp
    def read_dt(self):
        with open(self.path_dt, "rb") as f:
            _t = pickle.load(f)
        return _t

    def eqdt(self):
        if not self.hasdt():
            return False
        _t = self.read_dt()
        if _t == self.dt:
            return True
        else:
            return False
    def save(self, df):
        with open(self.path, "wb") as f:
            pickle.dump(df, f)
        with open(self.path_dt, "wb") as f:
            pickle.dump(self.dt, f)
    def delete(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        if os.path.exists(self.path_dt):
            os.remove(self.path_dt)

def excel_sheets(save_file, _tabs, sheet_name = "table", mode = None):
    if mode is None:
        if os.path.exists(save_file):
            mode = "a"
        else:
            mode = "w"
    with pd.ExcelWriter(save_file, mode = mode, engine = "openpyxl") as writer:
        _row = 0
        for i, _tab in enumerate(_tabs):
            _tab.to_excel(writer, sheet_name = sheet_name, startrow = _row, startcol = 0, index = False)
            _row += (2 + _tab. shape[0])


