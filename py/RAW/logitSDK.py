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
import pyecharts.options as opts

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
            recorder(self)
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
        _x = self.woevalue.loc[sample[0]]
        _y = self.Y.loc[sample[0]]
        _cols = pd.Series(cols.copy())
        _cols = _cols[_cols.isin(self.X.columns)]

        if labels is None:
            labels = ["set" + str(int(i + 1)) for i in range(len(sample))]
        self.train_cond = {labels[i]: sample[i] for i in range(len(sample))}

        while True:
            _x = _x[_cols]
            lrcv_L1 = LogisticRegression(C = C,
                                         penalty = mode,
                                         solver='liblinear',
                                         max_iter=100,
                                         class_weight = {0: 0.1, 1: 0.9})
            lrcv_L1.fit(_x, _y)
            lg_coef = pd.Series(lrcv_L1.coef_[0],index = _cols).sort_values()
            lg_coef = pd.DataFrame(lg_coef, columns=["Logistic"])
            lg_coef["Logistic"] = self.entL.loc[lg_coef.index] * lg_coef["Logistic"]
            exclud2 = lg_coef[lg_coef["Logistic"] <= rule]

            if len(exclud2) > 0:
                if step_wise:
                    exclud_index = exclud2[exclud2 == exclud2.min()]. index.tolist()
                else:
                    exclud_index = exclud2. index.tolist()
                _cols = pd.Series(_cols)[~pd.Series(_cols).isin(exclud_index)]
                continue
            else:
                cols = _cols.tolist()
                self.model = lrcv_L1
                self.model_cols = cols
                _x1 = self.woevalue.loc[:, cols]
                _score = pd.Series(lrcv_L1.predict_proba(_x1)[:, 1], index = _x1.index)
                _t = binning.\
                    tick(x = _score, quant = quant,
                         single_tick = False,
                         ruleV = _score.shape[0] / quant / 2,
                         mode = "v"). ticks_boundaries()
                self.score_ticks = _t
                self.model_result = model_result().from_model(self)
                res = self.model_result.binning_cond(pd.Series(True, index = self.Y.index))
                self.model_result_param = res
                return res

    def predict(self, X):
        X1 = self.trans_woe(cols = self.model_cols, X = X)[self.model_cols]
        return self.predict1(X1)

    def predict1(self, X1):
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
        _d = self.m.binning_tools.sub(self.m.model_cols)
        _d.cover_inf()
        res = dict()
        res["model"] = m.model
        res["cols"] = m.model_cols
        res["ticks"] = m.score_ticks
        res["standard_woe"] = m.standard_woe
        res["trans"] = _d.write_pattern()


        self.result = res
        return self.load_func()

    def load_func(self):
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
        x = self.m.woevalue[self.result["cols"]].loc[cond]
        y = self.m.Y.loc[cond]
        return self.binning_sample(x, y)

class sub_binning(OrderedDict):
    def write_cond(self, path = None):
        res = dict()
        for i, j in self.items():
            res1 = dict()
            res1["part"] = j.part
            res1["label"] = j.label
            res1["cond"] = j.cond
            res[i] = res1

        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(res, f)

        return res

    def write_pattern(self):
        res = dict()
        for i, j in self.items():
            res1 = dict()
            res1["data"] = j.write_pattern()
            res1["part"] = j.part
            res1["label"] = j.label
            res1["cond"] = j.cond
            res[i] = res1
        return res

    def write(self, path):
        _data = self.write_pattern()
        with open(path, "wb") as f:
            pickle.dump(_data, f)

    @staticmethod
    def read_pattern(data):
        res = sub_binning()
        for i, j in data.items():
            res1 = intLDict.read_pattern(j["data"])
            res1.part = j["part"]
            res1.label = j["label"]
            res[i] = res1
        return res

    @staticmethod
    def read(path):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return sub_binning.read_pattern(_d)

    def sub_info(self, i):
        _dfs = list()
        for k, j in self.items():
            _df = j[i]. result. copy()
            _df["label"] = j. label
            _df["part"] = j. part
            _df["porp"] = _df["cnt"] / _df["cnt"]. sum()
            _df["bad_rate"] = _df["mean"]
            _df.reset_name("bin", inplace = True)
            _df.reset_name("code", inplace = True)
            _dfs.append(_df)
        t_df = pd.concat(_dfs)
        return t_df

class recorder:
    home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    global_db = home + "/.model.db"

    def __init__(self, m, cwd = None):
        import os
        if cwd is None:
            cwd = os.getcwd()
        self.cwd = cwd
        self.m = m
        self.name = m.tsp
        self.m.recorder = self

        self.dir = "{0}/.model_profile/{1}/".  format(self.cwd, self.name)
        self.common_dir = "{0}/.model_profile/".  format(self.cwd)
        self.b_png_dir = self.dir + "b_png/"
        self.b_html_dir = self.dir + "b_html/"
        self.b_tool_dir = self.dir + "b_tool/"
        self.has_record = False

        for _path in [
                self.common_dir,
                self.dir,
                self.b_png_dir,
                self.b_html_dir,
                self.b_tool_dir,
        ]:
            if not os.path.exists(_path):
                os.mkdir(_path)

        self.db = "{0}/.model_profile/{1}/model.db".  format(self.cwd, self.name)
        self.common_db = "{0}/.model_profile/model.db".  format(self.cwd)
        self.conn = sqlite3.connect(self.db)
        self.conn_common = sqlite3.connect(self.common_db)
        self.conn_global = sqlite3.connect(self.global_db)

    @staticmethod
    def add_porp_lift(_b):
        _b = _b.copy()
        _b["porp"] = _b["cnt"] / _b["cnt"]. sum()
        _b["lift"] = (_b["mean"] / (_b["cnt"] * _b["mean"]).sum() * _b["cnt"]. sum()).fillna(0)
        _b.r1({"cnt": "总数",
               "porp": "占比",
               "lift": "提升",
               "mean": "坏率",
        })
        return _b

    def add_record(self, force = False):
        if force:
            self.has_record = False
        if not self.has_record:
            r = pd.DataFrame(pd.Series({
                "name": self.name,
                "db": self.db,
                "dir": self.dir,
                "start_dt": str(self.m.now),
                "sample_min_dt": self.m.X["dt"].  min(),
                "sample_max_dt": self.m.X["dt"].  max(),
                "sample_cnt": self.m.Y.shape[0],
                "columns_cnt": self.m.X.shape[1],
                "y_mean": self.m.Y.  mean(),
            })).T

            r.to_sql("records", self.conn_global, index = False, if_exists = "append")


        self.has_record = True

    def add_name(self, name):
        return "`{0}.{1}`".  format(self.name, name)

    def has_table(self, name):
        try:
            pd.read_sql("select * from {0} limit 1".  format(self.add_name(name)), self.conn)
            return True
        except:
            return False

    def save_table(self, df, name, append = False):
        if append:
            df.to_sql(self.add_name(name)[1: -1],
                      self.conn, index = False, if_exists = "append")
        else:
            self.drop_table(name)
            df.to_sql(self.add_name(name)[1: -1], self.conn, index = False)

    def drop_table(self, name):
        cursor = self.conn.cursor()
        try:
            cursor.execute("drop table {0}".  format(self.add_name(name)))
        except:
            pass

    def load_table(self, name):
        if self.has_table(name):
            return pd.read_sql("select * from {0}".  format(self.add_name(name)), self.conn)
        else:
            return None

    def save_df(self, df, name, dir = None):
        if dir is None:
            dir = self.dir
        with open(dir + name, "wb") as f:
            pickle.dump(df, f)

    def load_df(self, name):
        return pd.read_pickle(self.dir + name)

    @staticmethod
    def load_recorder():
        conn_global = sqlite3.connect(recorder.global_db)
        _df = pd.read_sql("select * from records", conn_global)
        return _df

    def load_html_path(self):
        return self.b_html_dir
    def load_html_files(self):
        return [self.b_html_dir + i for i in os.listdir(self.b_html_dir)]
    def load_html_map(self):
        _l = self.load_html_files()
        return {os.path.basename(i.split(".")[ - 2]):i for i in _l}

    def load_png_path(self):
        return self.b_png_dir
    def load_png_files(self):
        return [self.b_png_dir + i for i in os.listdir(self.b_png_dir)]
    def load_png_map(self):
        _l = self.load_png_files()
        return {os.path.basename(i.split("&")[0]):i for i in _l}

    def load_sub_binning_tools(self):
        _d = self.load_df("sub_binning_tools.pkl")
        return sub_binning.read_pattern(_d)
    def load_cols(self):
        return self.load_table("cols")
    def load_sub_cond(self):
        return self.load_df("sub_cond.pkl")
    def load_bifurcate(self):
        return self.load_table("bifurcate")

    def load_corr(self):
        return self.load_df("corr.pkl")
    def load_comment(self):
        return self.load_table("comment")

    def r_sample_dt(self):
        _z = pd.concat([self.m.X["dt"], self.m.Y], axis = 1)
        _z["dt"]
        _z["cut"] = pd.qcut(_z["dt"], 10)
        _df = _z.binning("cut", "label", func = lambda x, y, df:\
                         {"cnt": df.shape[0],
                          "mean": y.mean(),
                          "dt_min": df["dt"].  min(),
                          "dt_max": df["dt"].  max(),
                        }).sort_index()
        self.save_table(_df, "sample_dt")

    def r_binning_tools(self):
        _df = self.m.binning_tools.write_pattern()
        self.save_df(_df, "binning_tools.pkl")
        _l = list()
        for i, j in _df.items():
            _name = i
            _data = binList.read_pattern(j).write_pattern_json()
            _l.append([_name, _data])
        _l = pd.DataFrame(_l)
        _l.columns = ["name", "data"]
        self.save_table(_l, "binning")

    def r_sub_binning_tools(self):
        _df = self.m.sub_binning_tools.write_pattern()
        self.save_df(_df, "sub_binning_tools.pkl")
        _l = list()
        for i, j in _df.items():
            _label = j["label"]
            _part = j["part"]
            for k1, k2 in j["data"].items():
                _name = k1
                _data = binList.read_pattern(k2).write_pattern_json()
                _l.append([_label, _part, _name, _data])
        _l = pd.DataFrame(_l)
        _l.columns = ["label", "part", "name", "data"]
        self.save_table(_l, "sub_binning")
    def r_sub_cond(self):
        _df = self.m.sub_binning_tools.write_cond()
        self.save_df(_df, "sub_cond.pkl")
    def _binning_html1(self, name, path, UPPER = 0.08):
        _bt = self.m.binning_tools[name]
        x_label = _bt.info["bins"]
        #name = "card_cm_bank12m_pct"
        bar = Bar()
        bar.add_xaxis(xaxis_data=x_label)
        for l, s1 in self.m.sub_binning_tools.items():
            s = s1[name].  info
            try:
                bar.add_yaxis(series_name = l,
                              #category_gap = 0.2,
                              #gap = 0.1,
                              yaxis_data = (s["cnt"] / (s["cnt"].  sum())).tolist(),
                              label_opts = opts.LabelOpts(is_show = False),
                              itemstyle_opts = opts.ItemStyleOpts(opacity = 0.75)
                )
            except:
                bar.add_yaxis(series_name = l,
                              #category_gap = 0.2,
                              #gap = 0.1,
                              y_axis = (s["cnt"] / (s["cnt"].  sum())).tolist(),
                              label_opts = opts.LabelOpts(is_show = False),
                              itemstyle_opts = opts.ItemStyleOpts(opacity = 0.75)
                )
                
        ## 双y轴
        bar.extend_axis(
            yaxis=opts.AxisOpts(
                name="坏率",
                type_="value",
                min_=0,
            )
        ).set_global_opts(
            legend_opts = opts.LegendOpts(
                pos_top = "5%",
            ),

            title_opts = opts.TitleOpts(
                padding = 3,
                pos_left = "center",
            ),
            tooltip_opts=opts.TooltipOpts(
                is_show=True,
                trigger="axis",
                axis_pointer_type="cross"
            ),
            xaxis_opts = opts.AxisOpts(
                type_="category",
                ## x坐标标签 旋转
                axislabel_opts = opts.LabelOpts(rotate = 45,
                                                font_size = 10,
                                                position = "bottom",
                                                horizontal_align = "center",
                                                vertical_align = "middle",
                ),
                axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    type_="shadow"),
            ),
            yaxis_opts=opts.AxisOpts(
                name="占比",
                type_="value",
                min_=0,
                # max_=250,
                # interval=50,
                # axislabel_opts=opts.LabelOpts(formatter="{value} ml"),
                # axistick_opts=opts.AxisTickOpts(is_show=True),
                # splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )


        line = Line()
        line.add_xaxis(xaxis_data = x_label)
        for l, s1 in self.m.sub_binning_tools.items():
            s = s1[name].  info
            line.add_yaxis(
                linestyle_opts = opts.LineStyleOpts(
                    width = 3,
                    #type_ = "dashed",
                    type_ = "dotted",

                ),
                symbol_size = 6,
                series_name = l,
                yaxis_index = 1,
                y_axis = s["mean"].  apply(lambda x:min(x, UPPER)).  tolist(),
                label_opts=opts.LabelOpts(is_show=False)
            )
        bar.height = "350px"
        bar.width = "450px"
        grid = Grid()
        #grid.add(bar, grid_opts = opts.GridOpts(pos_top = '50%'))
        #grid.add(line, grid_opts = opts.GridOpts(pos_top = '50%'))
        bar.overlap(line)
        bar.render(template_name = "simple_chart_body.html", path = path)

    def r_binning_html(self):
        for i in self.m.binning_tools.keys():
            path = self.b_html_dir + i + ".html"
            self._binning_html1(i, path)

    def r_cmt(self):
        cmt = self.m.cmt.copy()
        cmt.name = "comment"
        cmt = cmt.reset_name("index")
        self.save_table(cmt, "comment")

    def r_xy(self):
        self.save_table(pd.concat([self.m.X, self.m.Y], axis = 1), "xy")

    def r_dtypes(self):
        _df = self.m.types_form
        self.save_table(_df, "dtypes")

    def r_bifurcate(self):
        _df = self.m.index_bifurcate.reset_name("index")
        self.save_table(_df, "bifurcate")

    def save_cols(self, cols, symbol = "test"):
        _df = pd.DataFrame([json.dumps(cols)])
        _df.columns = ["cols"]
        _df["dt"] = str(datetime.now())
        _df["symbol"] = symbol
        self.save_table(_df, "cols", append = True)

    def r_corr(self):
        _df = self.m.corr
        self.save_df(_df, "corr.pkl")

    def add_name_cmt(self, name, sep = "&", l1 = 100, l2 = 100):
        _cmt = self.m.cmt.get(name, "")
        if len(name) > l1:
            name = name[:l1] + "..."
        if len(_cmt) > l2:
            _cmt = _cmt[:l2] + "..."
        return name + sep + _cmt

    @staticmethod
    def _barplot(t_df,
                 title,
                 path,
                 draw_bad_rate = True,
                 upper_lim = None):

        import DRAW.draw
        t_df_copy = t_df.copy()
        if upper_lim is not None:
            t_df_copy["bad_rate"] = t_df_copy["bad_rate"]. apply(lambda x:min(x, upper_lim))
        DRAW.draw.draw_bar(t_df_copy,
                           title = title,
                           save = path,
                           draw_bad_rate = draw_bad_rate)

    def r_binning_png(self, cols = None, draw_bad_rate = True, upper_lim = 0.06):
        if cols is None:
            cols = list(self.m.sub_binning_tools.values().__iter__().__next__().keys())

        for i in cols:
            _name = (self.add_name_cmt(i) + ".png").replace("/", "_")
            _title = self.add_name_cmt(i, sep = "\n", l1 = 30, l2 = 20)
            path = self.b_png_dir + _name
            _df = self.m.sub_info(i)
            recorder._barplot(_df,
                              _title, path,
                              draw_bad_rate = draw_bad_rate,
                              upper_lim = upper_lim)

    def draw_corr_excel_addpng(self, path = None):
        if path is None:
            path = self.tsp + "_corrDrop.xlsx"
        import xlsxwriter
        d = self.var_drop
        _ent = self.entL
        workbook = xlsxwriter.Workbook(path)
        for i, j in d.items():
            cols = [i] + j.tolist()
            cols.sort(key = lambda x:_ent.get(x.split("&")[0], 0), reverse = True)
            worksheet = workbook.add_worksheet(i[:30])
            self.draw_excel_addpng(cols = cols, worksheet = worksheet)
        workbook.close()

        drop_var = [[i] + j.tolist() for i, j in self.var_drop.items()]
        drop_var1 = pd.DataFrame(pd.Series({j:k  for k, i in enumerate(drop_var) for j in i}, name = "part").sort_values())
        drop_var1["cmt"] = pd.Series(drop_var1.index).apply(lambda x:self.cmt.get(x, x)).values
        drop_var1 = drop_var1.reset_index()
        drop_var1 = drop_var1.set_index(["part", "index"])
        with pd.ExcelWriter(path, mode = "a", engine = "openpyxl") as writer:
            drop_var1. to_excel(writer, sheet_name = "total")

    def _excel_addpng(self, path, cols = None, sheet_name = "分箱图"):
        import DRAW.draw
        _d = self.load_png_map()
        if cols is None:
            cols = list(_d.keys())
        _ent = self.m.entL
        cols.sort(key = lambda x:_ent.get(x, 0), reverse = True)
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet(sheet_name)
        for i in range(len(cols)):
            _i1, _i2 = int(i / 3), i % 3
            worksheet.write(2 * _i1, 2 * _i2 + 1,"{0}".format(i))
            DRAW.draw.insert_image(worksheet,
                                   _d[cols[i]],
                                   row = 2 * _i1 + 1,
                                   col = 2 * _i2 + 1,
                                   x_scale = 3 ,
                                   y_scale = 2)
        workbook.close()

    def r_binning_excel(self, cols = None):
        self.binning_excel_path = self.dir + "binning.xlsx"
        self._excel_addpng(path = self.binning_excel_path, cols = cols)
        with pd.ExcelWriter(self.binning_excel_path, mode = "a", engine = "openpyxl") as writer:
            self.m.binL(). to_excel(writer, sheet_name = "全量",index = False)
            for i, j in self.m.sub_binL.items():
                j. to_excel(writer, sheet_name = i, index = False)

    def r_model_sample(self):
        path = self.dir + "modelsample.csv"
        self.sample_path = path
        self.m.oX[self.m.model_cols]. sample(min(1000, self.m.oX.shape[0])). to_csv(path, index = False)

    def r_model_result(self):
        path = self.dir + "modelresult.pkl"
        self.model_result_path = path
        self.m.model_result.save(path)

    def r_model_report(self):
        m = self.m
        mr = m.model_result
        _coef = mr.coef

        path = self.dir + "modelreport.xlsx"
        self.model_report_path = path

        try:
            self._excel_addpng(path, cols = _coef.index.tolist(),
                               sheet_name = "2.2分箱图")
        except:
            pass

        single_binning = m.binL().merge(m.cmt, left_on = "指标", right_index = True)
        single_binning["woe"] = single_binning["woe"] - m.standard_woe
        coef = pd.concat([_coef, m.cmt, m.entL], axis = 1, join = "inner").reset_name("指标名")
        b_sample = [mr.binning_cond(m.train_cond[i])["binning"]. reset_name(i) for i in sorted(m.train_cond)]
        result = dict()
        result["2.1单指标分箱"] = [single_binning]
        result["3.1模型参数"] = [coef]
        result["4.1样本分箱"] = b_sample

        for i in ["month", "channel"]:
            if i not in m.X.columns:
                continue
            _v = list(set(m.X[i]))
            _v.sort()
            _l1 = list()
            for k in _v:
                _cond = (m.X[i] == k)
                _res = mr.binning_cond(_cond)
                _l1.append(_res["binning"].reset_name(k))
            i1 = ("4.2月份分箱" if i == "month" else "5.1渠道分箱")
            result[i1] = _l1

        ## TODO excel表格中加上 KS通过_res["binning"]

        from default_var import excel_sheets
        for i in sorted(result.keys()):
            excel_sheets(path, result[i], i)

    def r_online(self, model_name, channel, container):
        """
        model_name = "自营天衍模型V1"
        channel = "ZY_TY_V1"
        container = "model_image1"
        """
        _dir = self.dir
        sample_path = _dir + "modelsample.csv"
        result_path = _dir + "modelresult.pkl"
        report_path = _dir + "modelreport.xlsx"

        os.system("cp {0} {1}.csv". format(sample_path, _dir + channel))
        os.system("cp {0} {1}.pkl". format(result_path, _dir + channel))
        os.system("cp {0} {1}模型报告.xlsx". format(report_path, _dir + channel))

        try:
            kw = dict()
            kw["model_name"] = model_name
            kw["channel"] = channel
            kw["container"] = container
            kw["date"] = datetime.now().strftime("%Y%m%d")
            kw["model_cols"] = "\n". join([str(i + 1) + ". " + j for i, j in enumerate(self.m.model_cols)])
            kw["model_cols_json"] = self.m.oX.iloc[0][self.m.model_cols]. to_dict()
            with open("/home/bozb/notebook/lsy/PYLIB/MODEL/FILE/online_file.txt", "r") as f:
                fs = f.read()
            with open(_dir + "{0}_{1}.txt". format(model_name, channel), "w") as f:
                f.write(fs.format(**kw))
        except:
            pass

class loader:
    home = recorder.home
    global_db = recorder.global_db

    def __init__(self, name, db, dir):
        self.name = name
        self.db = db
        self.dir = dir
        self.b_png_dir = self.dir + "b_png/"
        self.b_html_dir = self.dir + "b_html/"

        self.common_dir = os.path.dirname(self.dir[: -1]) + "/"
        self.common_db = self.common_dir + "model.db"

        import sqlite3
        self.conn = sqlite3.connect('{0}'.  format(self.db))
        self.conn_common = sqlite3.connect(self.common_db)
        self.conn_global = sqlite3.connect(self.global_db)

    add_name = recorder.add_name
    load_df = recorder.load_df
    has_table = recorder.has_table
    load_table = recorder.load_table
    save_table = recorder.save_table
    load_sub_binning_tools = recorder.load_sub_binning_tools
    load_sub_cond = recorder.load_sub_cond
    load_recorder = recorder.load_recorder

    load_html_path = recorder.load_html_path
    load_html_files = recorder.load_html_files
    load_html_map = recorder.load_html_map


    load_bifurcate = recorder.load_bifurcate
    load_cols = recorder.load_cols
    load_comment = recorder.load_comment
    save_cols = recorder.save_cols
    load_corr = recorder.load_corr

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
