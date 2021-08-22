import os
import pandas as pd
import math
import sqlite3
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import json
from RAW.int import binning, intLDict, intb
from RAW.ent import db_ent
from default_var import *

class index_selector:
    def __init__(self, path = None, log = None):
        assert not(path is None and log is None)
        import xlrd
        if path is None:
            path = log.flow_IO.binning_excel_path
        self.path = path
        data = xlrd.open_workbook(path) #打开demp.xlsx文件
        sheet_names = data.sheet_names()  #获取所有sheets名
        self.total_label = sheet_names[1]
        self.labels = sheet_names[2:]
        info = dict()
        for j, i in enumerate(sheet_names[1:]):
            d = pd.read_excel(path, sheet_name = i)
            d1 = \
            {j1: j2.set_index("number") for j1, j2 in d.groupby("指标", as_index = False)}
            if j == 0:
                self.total = d1
            else:
                info[i] = d1
        self.info = info
        self.entL = pd.Series({i:j["区分度"]. iloc[0] for i, j in self.total.items()})
        mono_info = pd.Series({i: self.is_mono(i) \
                    for i in self.total.keys()}).sort_values(ascending = False)
        self.mono = mono_info
        if log is not None:
            self.X = log.X
            self.Y = log.Y
            self.corr = log.corr
            self.corr_ind = pd.Series(self.corr.index)
            self.std = self.X[self.mono[self.mono]. index]. std()

    def mean_dif_ent(self, index):
        from RAW.ent import db_ent
        infos = []
        std_rate = self.total[index]["bad_cnt"]. sum() / self.total[index]["总数"]. sum()
        for l, k in enumerate(self.labels):
            _df = self.info[k][index]
            m = _df["总数"]
            info = _df[[]]
            info["label"] = k
            info.std_rate = _df["bad_cnt"]. sum() / _df["总数"]. sum()
            add_cnt = 1 / info.std_rate
            info["bad_rate"] = (_df["bad_cnt"] + 1) / (_df["总数"] + add_cnt)
            infos.append(info)
            if l == 0:
                m0 = pd.Series(1, index = m.index)
            m0 *= m
        m0 = pd.Series(m0 ** (1 / len(self.labels)), name = "cnt")
        for i in infos:
            i["cnt"] = m0
            i["bad"] = i["bad_rate"] * i["cnt"]
            i["good"] = i["cnt"] - i["bad"]

        infos = pd.concat(infos).reset_index()
        def ent_diff(infos, i, j):
            ma1 = infos[infos["number"]. isin([i])]. set_index("label")[["bad", "good"]]
            ma2 = infos[infos["number"]. isin([j])]. set_index("label")[["bad", "good"]]
            ma3 = ma1 + ma2
            _dif = (db_ent(ma1.values) + db_ent(ma2.values) - db_ent(ma3.values))
            return _dif
        bt = self.total[index]
        distance = pd.DataFrame(1, columns = m0.index, index = m0.index)
        distance = distance - pd.DataFrame(np.eye(distance.shape[0]))
        ent_matrix = pd.DataFrame(0, columns = m0.index, index = m0.index)
        continues_span = []
        l_begin = 0
        for i in range(bt.shape[0]):
            if bt.loc[i, "is_special"]:
                l_end = i - 1
                l_begin = i + 1
            else:
                if i == (bt.shape[0] - 1):
                    l_end = i
                else:
                    continue
            if l_end >= l_begin:
                continues_span.append([l_begin, l_end])

        total_cnt = m0.sum()
        for begin, end in continues_span:
            m1 = m0[begin:(end + 1)]
            m2 = m1. cumsum()
            m3 = m2 - m1 / 2
            for k1 in range(begin, end + 1):
                for k2 in range(begin, end + 1):
                    distance.loc[k1, k2] = 1 - abs(m3[k1] - m3[k2]) / total_cnt

        for k1 in range(m0.shape[0]):
            for k2 in range(m0.shape[0]):
                if k1 != k2:
                    ent_matrix.loc[k1, k2] = ent_diff(infos, k1, k2)

        t_ent = (((ent_matrix * m0).T * m0) * distance).sum().sum() / ((m0.sum())**2)
        return t_ent

    def porp_dif_ent(self, index):
        infos = []
        for l, k in enumerate(self.labels):
            m = self.info[k][index]["总数"]
            infos.append(m)
        _m = pd.concat(infos, axis = 1)
        return db_ent(_m.values)

    def var_find(self):
        b = dict()
        b[0] = pd.Series({i: self.mean_dif_ent(i) \
                        for i in self.total.keys()}).sort_values(ascending = False)
        b[1] = pd.Series({i: self.porp_dif_ent(i) \
                        for i in self.total.keys()}).sort_values(ascending = False)
        b[2] = self.entL.sort_values(ascending = True)
        self.bi = b
        return b

    def is_mono(self, index):
        _res = self.total[index]
        if _res["type"]. iloc[0] == 'str':
            return False
        _res_dif = _res.loc[(_res["is_special"]) == False]["坏率"]. diff().iloc[1:]
        _mono = (_res_dif > 0).sum()*(_res_dif < 0).sum() == 0
        return _mono

    def cor_bet(self, i, t1 = 0.5, t2 = 0.95):
        if isinstance(i, str):
            return self.corr[i][self.corr[i]. between(t1, t2)]. sort_values()
        return pd.concat([self.cor_bet(i1, t1, t2) for i1 in i], axis = 1, join = "inner")

    def roc_combine(self, main_index, additive_coef):
        additive_coef_adj = additive_coef / self.std[additive_coef.index] * self.std[main_index]
        _new_x = self.X[main_index] + self.X[additive_coef_adj.index]@additive_coef_adj
        return {"auc": roc_auc_score(self.Y, _new_x),
                "x": _new_x,
                "coef": additive_coef,
                "coef_adj": additive_coef_adj, }

    def roc_combine1(self, main_index, additive_coef):
        _new_x = self.X[main_index] + self.X[additive_coef.index]@additive_coef
        return roc_auc_score(self.Y, _new_x)

    def roc_route(self, main_index,
                  additive_coef = None,
                  cols_range = None,
                  alpha = 0.001,
                  t1 = 0.5,
                  t2 = 0.95,
                  std_dif = 3,
    ):
        if additive_coef is None:
            additive_coef = pd.Series([])
        additive_coef = additive_coef.copy()
        mono_cols = self.mono[self.mono]. index
        _roc = self.roc_combine(main_index, additive_coef)
        route = [self.roc_combine(main_index, additive_coef)]
        main_std = self.std[main_index]
        std_cols = self.std[self.std.between(main_std / std_dif, main_std * std_dif)]. index
        while True:
            additive_coef = additive_coef[additive_coef != 0]
            cur_cols = additive_coef.index.tolist()
            selected_cols = [main_index] + cur_cols
            print("select", selected_cols)
            add_cols = self.cor_bet(selected_cols, t1, t2).index.tolist()
            print(t1, t2)
            print("add", add_cols)

            if cols_range is not None:
                add_cols = list(set(add_cols) & set(cols_range))
            next_selections = []
            print(add_cols)
            for _col in cur_cols + add_cols:
                if _col not in mono_cols:
                    continue
                if _col not in std_cols:
                    continue
                for _coef in [-0.1, 0.1]:
                    _c = additive_coef. copy()
                    _c[_col] = additive_coef.get(_col, 0) + _coef
                    _c = _c[_c != 0]
                    if (_c.max() > 0.55) | (_c.min() < -0.35):
                        continue
                    if (_c.shape[0] >= 3.5):
                        continue
                    if (_c[_c >= 0]. sum()) >= 0.95:
                        continue
                    if (_c[_c <= 0]. sum()) <= -0.55:
                        continue
                    next_selections.append(_c)
            if len(next_selections) == 0:
                break
            res = pd.DataFrame([self.roc_combine(main_index, i) \
                                for j, i in enumerate(next_selections)])
            s = (res["auc"] - 0.5).abs() - res["coef"]. apply(lambda x:len(x)) * alpha
            _ind = s[s == s.max()]. index[0]
            _route_add = res.loc[_ind]
            additive_coef = _route_add["coef"]
            if (abs(_route_add["auc"] - 0.5) - alpha * len(_route_add["coef"])) \
               <= (abs(route[ - 1]["auc"] - 0.5) + alpha - alpha * len(route[ - 1]["coef"])):
                break
            route.append(_route_add)
        return route

            
