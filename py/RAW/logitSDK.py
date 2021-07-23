import pickle
import math
import os, sys
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import json
import sqlite3
import xlsxwriter
from shutil import copyfile
from pyecharts.charts import Bar, Line, Grid
from .cluster import col_cluster
import pyecharts.options as opts
import pandas as pd
from .recorder import recorder, loader, sub_binning
## sys path 添加py所在目录
try:
    _tmp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))[: -1])
except:
    _tmp_dir = os.path.dirname(os.getcwd())

sys.path.append(_tmp_dir)
_root_dir = os.path.dirname(_tmp_dir)
from RAW.int import *
from RAW.ent import db_ent
from default_var import *

## simple_chart_body.html 作为模板加入到 pyecharts 的模板库中
try:
    import pyecharts
    echart_root = pyecharts.__path__[0]
    echart_templates = echart_root + "/render/templates"
    assert os.path.exists(echart_templates)
    copyfile(_root_dir + "/simple_chart_body.html",
             echart_templates + "/simple_chart_body.html"
    )
except:
    raise

def cond_part(_dt, _l):
    """
    将_dt以_l作为分为数切割
    返回掩码列
    """
    if isinstance(_l, float):
        _l = [_l]
    assert isinstance(_l, list)
    assert len(_l) >= 1
    _cond = list()
    _dts = _dt.quantile(_l).tolist()
    _cond.append(_dt <= _dts[0])
    for i in range(len(_l) - 1):
        _cond.append((_dt <= _dts[i + 1]) & (_dt > _dts[i]))
    _cond.append(_dt > _dts[ - 1])
    return [_c.values for _c in _cond]

def KS(y, x):
    z = pd.concat([y, x], axis = 1)
    z.columns = ["label", "x"]
    z = z.sort_values("x")
    z_bad = z["label"]. cumsum() / (y == 1).sum()
    z_good = (z["label"] == 0). cumsum() / (y == 0).sum()
    return - (z_bad - z_good).min()


## TODO 改写为 引用 fit_func
def lazy_fit_func(self, x, y):
    """
    fit function的惰性形态
    优先从self中取,如果已存则可以省去列计算的代价
    如果self中未发现,则按正常逻辑计算
    """
    _res = dict()
    if hasattr(self, "cnt"):
        _res["cnt"] = self.cnt
    else:
        _res["cnt"] = y.shape[0]

    if hasattr(self, "woe"):
        _res["woe"] = self.woe
    else:
        _res["woe"] = math.log(((y == 1).sum() + 0.5) / ((y == 0).sum() + 0.5))

    if hasattr(self, "mean"):
        _res["mean"] = self.mean
    else:
        _res["mean"] = y.mean()

    if hasattr(self, "is_special"):
        _res["is_special"] = self.is_stick()
    else:
        _res["is_special"] = self.is_stick()

    if hasattr(self, "good_cnt"):
        _res["good_cnt"] = self.good_cnt
    else:
        _res["good_cnt"] = (y == 0).sum()

    if hasattr(self, "bad_cnt"):
        _res["bad_cnt"] = self.bad_cnt
    else:
        _res["bad_cnt"] = (y == 1).sum()
    return _res

def step_train(x, y, ent, C, rule=0, mode="l1", step_wise=True):
    _cols = pd.Series(x.columns).copy()
    while True:
        x = x[_cols]
        lrcv_L1 = LogisticRegression(C = C,
                                     penalty = mode,
                                     solver='liblinear',
                                     max_iter=100,
                                     class_weight = {0: 0.1, 1: 0.9})
        lrcv_L1.fit(x, y)
        lg_coef = pd.Series(lrcv_L1.coef_[0],index = _cols).sort_values()
        lg_coef = pd.DataFrame(lg_coef, columns=["Logistic"])
        lg_coef["Logistic"] = ent.loc[lg_coef.index] * lg_coef["Logistic"]
        exclud2 = lg_coef[lg_coef["Logistic"] <= rule]

        if len(exclud2) > 0:
            if step_wise:
                exclud_index = exclud2[exclud2 == exclud2.min()]. index.tolist()
            else:
                exclud_index = exclud2. index.tolist()
            _cols = pd.Series(_cols)[~pd.Series(_cols).isin(exclud_index)]
            continue
        else:
            return {"cols": _cols.tolist(), "model": lrcv_L1}

class lgt:
    default_kwargs = {
        "mode": "b",
        "ruleV": 1000,
        "ruleB": 10,
        "ruleM": 1,
        "ruleC": -0.0001,
        "quant": 30,
    }

    def X_transform(X, trans_rule, trans_v):
        for i, j in trans_v.items():
            globals()[i] = j
        for i in X.columns:
            if i in trans_rule:
                print("X[i]" + trans_rule[i])
                X[i] = eval("X[i]" + trans_rule[i]).values

        if "total" in trans_rule:
            print("X" + trans_rule["total"])
            X = eval("X" + trans_rule["total"])
        return X


    def __init__(self,
                 X,
                 Y,
                 cmt = None,
                 trans_rule = {"total": ".replace(\'\', -999999).fillna(-1)",
                 },
                 trans_v = dict(),
                 record = True,
                 ** model_v,
                 ):

        #add_guests("lgt_init.txt")
        self.now = datetime.now()

        # 1.sample must have datetime information
        assert ("dt" in X.columns)

        # 2.keep the original data pointer
        self.oX = X
        self.oY = Y
        X = X.reset_index(drop = True)
        Y = Y.reset_index(drop = True)

        # 3.preprocess transform
        X["month"] = pd.to_datetime(X["dt"]).dt.strftime("%Y-%m")
        X = lgt.X_transform(X = X, trans_rule = trans_rule, trans_v = trans_v)

        # 4.type transform str->float
        dtypes = X.dtypes.astype(str)
        dtypes1 = pd.Series()
        dtypes2 = dtypes.copy()

        _trans_type_dict = dict()
        for i in dtypes.index:
            t = dtypes[i]
            if t != "object":
                continue
            i1, i2 = binning.guess_type(X[i])
            if i1 == "int":
                print("as float", i)
                _trans_type_dict[i] = i2
                dtypes1[i] = "float64"
                dtypes2[i] = "float64"

        if len(_trans_type_dict) > 0:
            _X2 = pd.DataFrame(_trans_type_dict)
            X.drop(list(_trans_type_dict.keys()), axis = 1, inplace = True)
            X = pd.concat([X, _X2], axis = 1)

        self.original_types = dtypes
        self.trans_types = dtypes1
        self.new_types = dtypes2
        self.types_form = pd.concat([self.original_types,
                                     self.trans_types,
                                     self.new_types
        ], axis = 1)
        self.types_form.columns = ["original_types", "trans_types", "new_types"]

        # 5.set attributes x, y, cmt...
        self.X = X
        self.Y = Y
        self.Y.name = "label"

        if cmt is None:
            cmt = pd.Series(X.columns, index = X.columns, name = "comment")
        else:
            cmt = cmt. cv(self.X.columns)
        self.cmt = cmt
        self.cmt.name = "注释"

        for i, j in model_v.items():
            setattr(self, i, j)

        self.init_tsp()
        if record:
            recorder.from_m(self)
        self.init_bintool()
        self.init_basic()


    def binning_cnt(self):
        return pd.Series({i: len(j) for i, j in self.binning_tools.items()})

    def init_basic(self):
        self.standard_woe = math.log(((self.Y == 1).sum() + 0.5) / ((self.Y == 0).sum() + 0.5))

    def init_tsp(self):
        self.tsp = "Lgt" + datetime.now().strftime("%Y_%m%d_%H%M_%s")

    def init_bintool(self):
        if not hasattr(self, "binning_tools"):
            self.binning_tools = intLDict()
        if not hasattr(self, "binning_result"):
            self.binning_result = dict()
        if not hasattr(self, "error_inds"):
            self.error_inds = set()

    def tick(self, cols = None, pass_error = True, **kwargs):
        """
        'tick' include 'fit'
        """
        kwargs["func"] = intb.fit_func
        self.init_bintool()
        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        for i in cols:
            kwargs["x"] = self.X[i]
            try:
                self.binning_tools[i] = binning.tick(**{**lgt.default_kwargs, **kwargs})
            except Exception as e:
                if pass_error:
                    self.error_inds.add(i)
                    continue
                else:
                    raise e

    def fit(self, cols = None, pass_error = True, **kwargs):
        """
        fit: update the mean, cnt, etc. params using X ans Y in kwargs
        """
        kwargs["func"] = intb.fit_func
        self.init_bintool()
        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        for i in cols:
            kwargs["x"] = self.X[i]
            try:
                self.binning_tools[i]. calculate(**kwargs)
            except Exception as e:
                if pass_error:
                    self.error_inds.add(i)
                    continue
                else:
                    raise e

    def binning(self, cols = None, pass_error = True, remain_tick = True, fit_mode = "b", **kwargs):
        """
        mode 拆分为 mode 和 fit_mode
        mode 决定分箱的类型 刻度 info的指标
        fit_mode 决定result选取哪些指标
        """

        ## if fit_mode == "b":
        ##     fit_func = lazy_fit_func
        ## if fit_mode == "c":
        ##     fit_func = intc.fit_func

        fit_func = lazy_fit_func

        self.init_bintool()
        #if fit_func is None:
        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        for i in cols:
            kwargs["x"] = self.X[i]

            _remain_tick = False
            if remain_tick:
                if i in self.binning_tools:
                    _remain_tick = True
            try:
                if _remain_tick == False:
                    self.binning_tools[i] = \
                        binning.binning(
                            **{**lgt.default_kwargs,
                               **kwargs,
                               "func": fit_func})
                else:
                    self.binning_tools[i]. result = \
                        self.binning_tools[i]. calculate(
                            **{**lgt.default_kwargs,
                               **kwargs,
                               "func": fit_func})

            except Exception as e:
                if i in self.binning_tools:
                    del self.binning_tools[i]
                if pass_error:
                    self.error_inds.add(i)
                    continue
                else:
                    raise e


    def reset_info(self):
        for i in self.binning_tools.values():
            i.reset_info()

    def change_xy(self, x, y):
        _items = ["cmt",
                  "binning_tools",
                  "binning_result",
        ]
        _kw = deepcopy({i: self.__dict__[i] for i in _items})
        _t = lgt(X = x, Y = y, record = False, **_kw)
        _t.reset_info()
        _t.fit()
        return _t

    def change_xy_cond(self, cond):
        _t = self.change_xy(x = self.X.loc[cond], y = self.Y.loc[cond])
        _t.binning_tools.cond = cond
        return _t

    def subobj_condition(self, conds, labels):
        _l = list()
        for i in range(len(conds)):
            _o = self.change_xy_cond(conds[i])
            _o.label = labels[i]
            _o.part = int(i + 1)
            _o.binning_tools.label = _o.label
            _o.binning_tools.part = _o.part
            _o.binning_tools.cond = conds[i]
            _l.append(_o)
        return _l

    def draw_binning(self,
                     conds,
                     cols = None,
                     labels = None,
                     draw_bad_rate = True,
                     upper_lim = None
                    ):
        if labels is None:
            labels = [str(i + 1) for i in range(len(conds))]
        self.binning_labels = labels
        if cols is None:
            cols = self.binning_tools.keys()
        subs = self.subobj_condition(conds = conds, labels = labels)
        _ = [i.fit(cols = cols) for i in subs]
        _ = [i.binning(cols = cols) for i in subs]

        self.sub_binL = OrderedDict()
        self.sub_binning_tools = sub_binning()

        for i in subs:
            self.sub_binL[i.label] = i.binL()
            self.sub_binning_tools[i.label] = i.binning_tools

        return subs

    def sub_info(self, i):
        return self.sub_binning_tools.sub_info(i)

    def trans_type(self, i, X = None):
        if X is None:
            X = self.X
        return self.binning_tools[i]. fit_type(X[i])

    def trans(self, i, keyword, X = None):
        if X is None:
            X = self.X
        _res = self.binning_tools[i]. trans(self.trans_type(i, X), keyword)
        if keyword == "woe":
            _res = _res - self.standard_woe
        return _res

    def trans_woe(self, cols = None, X = None):
        if X is None:
            X = self.X
        if cols is None:
            cols = list(self.binning_tools.keys())
        _df = pd.DataFrame([self.trans(i, "woe", X) for i in cols]).T
        return _df

    def trans_bin(self, cols = None, X = None):
        if X is None:
            X = self.X
        if cols is None:
            cols = list(self.binning_tools.keys())
        _df = pd.DataFrame([self.trans(i, "word", X) for i in cols]).T
        return _df

    def ent(self, i):
        return db_ent(self.binning_tools[i]. result[["bad_cnt", "good_cnt"]]. values)

    @property
    def entL(self):
        return pd.Series({i:self.ent(i) for i in self.binning_tools.keys()}, name = "ent").sort_values(ascending = False)

    def binL(self):
        _ent = self.entL
        _dfs = list()
        for i in _ent.index:
            _df = recorder.add_porp_lift(self.binning_tools[i].\
                                    result.reset_name("区间").reset_name("number"))
            _df["指标"] = i
            _df["区分度"] = _ent.loc[i]
            _df["type"] = "int" if "intList" in str(type(self.binning_tools[i])) else "str"
            _dfs.append(_df)
        return pd.concat(_dfs)

    def init_woe_info(self):
        if not hasattr(self, "woevalue"):
            self.woevalue = pd.DataFrame(index = self.X.index)

    def update_woe(self, cols = None):
        self.init_woe_info()
        if cols is None:
            cols = self.binning_tools.keys()
        for i in cols:
            self.woevalue[i] = self.trans(i, "woe").astype(float)
        self.corr = self.woevalue.corr()

    def cluster(self, cols, stop_dist=0.7, stop_nodecnt=2):
        from cluster import Hierarchical
        DISTANCE = (1 - (self.corr.loc[cols, cols])**2)
        h = Hierarchical(data=DISTANCE)
        names = [(i, ) for i in cols]
        cluster_res = h.hcluster(col_list=names, stop_dist=0.95, stop_nodecnt=3)
        cluster_res1 = [self.entL.loc[list(i)] for i in cluster_res]
        cluster_res1.sort(key=lambda x:x.max(), reverse=True)
        return cluster_res1

    def var(self, cols = None, exclud = [], includ = [], rule = 0.6):
        _bcnt = self.binning_cnt()
        exclud += _bcnt[_bcnt == 1]. index.tolist()
        if cols is None:
            cols = self.binning_tools.keys()
        _ent1 = self.entL.loc[cols]. copy()
        _ent1 = _ent1[~_ent1.index.isin(exclud)]. sort_values(ascending = False)
        _corr1 = self.corr.loc[_ent1.index, _ent1.index]. copy()

        _drop_dict = dict()
        for tmp_ind in includ:
            drop_cols = _corr1.index[_corr1.loc[:, tmp_ind] > rule]
            drop_cols = drop_cols[~drop_cols.isin([tmp_ind] + includ)]
            _drop_dict[tmp_ind] = drop_cols
            _ent1.drop(drop_cols, inplace = True)
            _corr1 = _corr1.drop(drop_cols, axis = 1).drop(drop_cols, axis = 0)
        i = 0
        while True:
            if i >= _ent1.shape[0]:
                break
            tmp_ind = _ent1.index[i]
            drop_cols = _corr1.index[_corr1.loc[:, tmp_ind] > rule]
            drop_cols = drop_cols[~drop_cols.isin([tmp_ind] + includ)]
            _drop_dict[tmp_ind] = drop_cols
            _ent1.drop(drop_cols, inplace = True)
            _corr1 = _corr1.drop(drop_cols, axis = 1).drop(drop_cols, axis = 0)
            i += 1
        self.var_drop = _drop_dict
        return _ent1

    def train(self, cols,
              sample,
              labels = None,
              rule = 0,
              C = 0.5,
              step_wise = True,
              mode = "l1",
              quant = 10,
    ):
        if labels is None:
            labels = ["set" + str(int(i + 1)) for i in range(len(sample))]
        self.train_cond = {labels[i]: sample[i] for i in range(len(sample))}
        
        _x = self.woevalue.loc[sample[0], cols]
        _y = self.Y.loc[sample[0]]
        _ent = self.entL
        _res = step_train(_x, _y, C=C, rule=rule, ent=_ent, mode=mode, step_wise=step_wise)

        model = _res["model"]
        cols = _res["cols"]
        _x1 = self.woevalue.loc[:, cols]
        _score = pd.Series(model.predict_proba(_x1)[:, 1], index = _x1.index)
        _t = binning.\
            tick(x = _score, quant = quant,
                 single_tick = False,
                 ruleV = _score.shape[0] / quant / 2,
                 mode = "v"). ticks_boundaries()
        self.model = model
        self.model_cols = cols
        self.score_ticks = _t
        self.model_result = model_result().from_model(self)
        res = {labels[i]:self.model_result.binning_cond(sample[i]) for i in range(len(sample))}
        self.model_result_param = res
        return res        

    def predict(self, X):
        """
        woe变换+LR模型
        """
        X1 = self.trans_woe(cols = self.model_cols, X = X)[self.model_cols]
        return self.predict1(X1)

    def predict1(self, X1):
        """
        纯LR模型
        """
        X1 = X1[self.model_cols]
        return pd.Series(self.model.predict_proba(X1)[:, 1], index = self.X.index, name = "score")

    @staticmethod
    def load_model_result(path):
        RES = pd.read_pickle(path)
        RES["trans"] = intLDict.read_pattern(RES["trans"])
        RES["trans_func"] = lambda x:RES["trans"]. trans(x[RES["cols"]], default = RES["standard_woe"]) - RES["standard_woe"]
        return RES

    def var_find(self, path = None, draw = False):
        if path is None:
            path = self.recorder.binning_excel_path
        be = binning_excel(log = self)
        b = be.var_find()
        b1 = pd.concat(b, axis = 1)
        b1.columns = ["bif_mean", "bif_porp", "bif_ent"]
        self.index_bifurcate = b1

        if draw:
            for i in range(2):
                _b = b[i].index.tolist()
                _b_filename = self.tsp + "_b{0}.xlsx". format(str(i))
                workbook = xlsxwriter.Workbook(_b_filename)
                worksheet = workbook.add_worksheet("compare")
                self.draw_excel_addpng(cols = _b, worksheet = worksheet)
                workbook.close()
        return b

class model_result:
    """
    object style
    """
    def from_model(self, m):
        self.m = m
        self.X = self.m.woevalue
        self.Y = self.m.Y        
        _d = self.m.binning_tools.sub(self.m.model_cols)
        _d.cover_inf()
        res = dict()
        res["model"] = m.model
        res["cols"] = m.model_cols
        res["ticks"] = m.score_ticks
        res["standard_woe"] = m.standard_woe
        res["trans"] = _d.write_pattern()
        res["cond"] = m.train_cond
        self.result = res
        return self.load_func()

    def from_result(self,
                    model,
                    cols,
                    ticks,
                    standard_woe,
                    binning_tools,
                    train_cond=None, 
                    lgt=None
    ):
        res = dict()
        res['model'] = model
        res['cols'] = cols
        res['ticks'] = ticks
        res['standard_woe'] = standard_woe
        res['trans'] = binning_tools.sub(cols).write_pattern()
        res['cond'] = train_cond
        self.result = res
        self.m = lgt
        return self.load_func()

    def load_xy(self, x, y):
        self.X = x
        self.Y = y
        return self

    def load_func(self):
        """
        加载woe转换函数
        """
        res = self.result
        self.tools = intLDict.read_pattern(res["trans"])
        self.func = lambda x:self.tools.\
            trans(x[res["cols"]],
                  default = res["standard_woe"]) - res["standard_woe"]
        self.coef = pd.Series(
            res["model"].coef_[0],
            index = res["cols"],
            name = "系数")
        return self

    def load_pattern(self, res):
        self.result = res
        return self.load_func()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.result, f)
        return self

    def binning_sample(self, x, y):
        res = self.result
        cols = res["cols"]
        _score = pd.Series(res["model"].predict_proba(x[cols])[:, 1], index = x.index)
        z = pd.concat([y, _score], axis = 1)
        z.columns = ["label", "x"]
        _b = z.binning(x = "x", y = "label", l = res["ticks"])

        res = dict()
        res["KS"] = KS(y, _score)
        res["AUC"] = roc_auc_score(y, _score)
        res["binning"] = recorder.add_porp_lift(_b)
        return res


    def binning_cond(self, cond):
        x = self.X[self.result["cols"]].loc[cond]
        y = self.Y.loc[cond]
        return self.binning_sample(x, y)

class binning_excel:
    def __init__(self, path = None, log = None):
        assert not(path is None and log is None)
        import xlrd
        if path is None:
            path = log.recorder.binning_excel_path
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

if __name__ == "__main__":
    pass
