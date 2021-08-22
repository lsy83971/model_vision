from collections import defaultdict
import os
import pandas as pd
import math
import sqlite3
import pickle
from datetime import datetime
from collections import OrderedDict
from shutil import copyfile
from pyecharts.charts import Bar, Line, Grid
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import json
from RAW.int import binning, intLDict, intb
from RAW.ent import db_ent
from default_var import *
from copy import deepcopy
from functools import partial, partialmethod
from .index_selector import index_selector

from pyecharts.charts import Bar, Line, Grid
import pyecharts.options as opts


try:
    _tmp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))[: -1])
except:
    _tmp_dir = os.path.dirname(os.getcwd())

sys.path.append(_tmp_dir)
_root_dir = os.path.dirname(_tmp_dir)

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

def echarts_plot(x_label, info, UPPER=0.05):
    info = OrderedDict(info)
    bar = Bar()
    bar.add_xaxis(xaxis_data=x_label)
    for l, s in info.items():
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
    for l, s in info.items():
        line.add_yaxis(
            linestyle_opts = opts.LineStyleOpts(
                width = 3,
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
    bar.overlap(line)    
    return bar



## TODO 改写为 引用 fit_func
def lazy_fit_func(self, x, y):
    """
    fit function的惰性形态
    优先从self中取,如果已存则可以省去列计算的代价
    如果self中未发现,则按正常逻辑计算
    zz
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

class sub_binning(OrderedDict):
    
    def save_cond(self, path = None):
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

    def save_format(self):
        res = dict()
        for i, j in self.items():
            res1 = dict()
            res1["data"] = j.save_format()
            res1["part"] = j.part
            res1["label"] = j.label
            res1["cond"] = j.cond
            res[i] = res1
        return res

    def save(self, path):
        _data = self.save_format()
        with open(path, "wb") as f:
            pickle.dump(_data, f)

    @staticmethod
    def load_format(data):
        res = sub_binning()
        for i, j in data.items():
            res1 = intLDict.load_format(j["data"])
            res1.part = j["part"]
            res1.label = j["label"]
            res[i] = res1
        return res

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return sub_binning.load_format(_d)

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

    
class model_flow:
    
    default_kwargs = {
        "mode": "b",
        "ruleV": 1000,
        "ruleB": 10,
        "ruleM": 1,
        "ruleC": -0.0001,
        "quant": 30,
    }
    
    @staticmethod
    def X_transform(X, trans_rule, trans_v):
        """
        transform X by columns or uniformly.
        trans_rule : column_name -> '.replace(...).fillna(...)'
                         'total' -> '.replace(...).fillna(...)'
        trans_v : variable used in trans_rule
        """
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

    def from_data(
            self,
            X,
            Y,
            cmt = None,
            trans_rule = {"total": ".replace(\'\', -999999).fillna(-1)",
            },
            trans_v = dict(),
            record = True,
            ** model_v,
    ):
        """
        initialize the model_flow object from data
        the only entrance to CREATE a NEW MODEL_FLOW
        """
        # I.*************** load XY information **************
        
        # 1.X must have 'dt' in columns
        
        assert ("dt" in X.columns)
        
        # 2.keep pointer to the original data
        
        self.oX = X
        self.oY = Y
        
        # 3.preprocess transform with `trans_rule` and `trans_v`
        
        X = X.reset_index(drop = True)
        Y = Y.reset_index(drop = True)
        Y.name = "label"        
        X["month"] = pd.to_datetime(X["dt"]).dt.strftime("%Y-%m")
        X = model_flow.X_transform(X = X, trans_rule = trans_rule, trans_v = trans_v)
        
        # 4.try to transform dtypes `str` to `float`
        
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

        types_form = pd.concat([dtypes,
                                dtypes1,
                                dtypes2
        ], axis = 1)
        types_form.columns = ["original_types", "trans_types", "new_types"]
        standard_woe = math.log(((Y == 1).sum() + 0.5) / ((Y == 0).sum() + 0.5))

        if cmt is None:
            cmt = pd.Series(X.columns, index = X.columns, name = "comment")
        else:
            cmt = cmt. cv(X.columns)
        cmt.name = "注释"            
        
        self.X = X
        self.Y = Y
        self.Y.name = "label"
        self.types_form = types_form
        self.standard_woe = standard_woe            
        self.cmt = cmt
        
        # II.*************** init basic tools **************


        self.binning_tools = intLDict()        
        self.error_inds = set()

        for i, j in model_v.items():
            setattr(self, i, j)
        return self

    def tick(self, cols = None, pass_error = True, **kwargs):
        """
        'tick' include 'fit'
        """
        binning_tools = self.binning_tools
        error_inds = self.error_inds
        
        kwargs["func"] = intb.fit_func
        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        for i in cols:
            kwargs["x"] = self.X[i]
            try:
                binning_tools[i] = binning.tick(
                    **{**model_flow.default_kwargs, **kwargs})
            except Exception as e:
                if pass_error:
                    error_inds.add(i)
                    continue
                else:
                    raise e

        self.error_inds = error_inds
        self.binning_tools = binning_tools

    def fit(self, cols = None, pass_error = True, **kwargs):
        """
        fit: update the mean, cnt, etc. 
        params calculated from X and Y in kwargs
        """
        binning_tools = self.binning_tools
        error_inds = self.error_inds        
        
        kwargs["func"] = intb.fit_func
        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        for i in cols:
            kwargs["x"] = self.X[i]
            try:
                self.binning_tools[i]. calculate(**kwargs)
            except Exception as e:
                if pass_error:
                    error_inds.add(i)
                    continue
                else:
                    raise e

        self.binning_tools = binning_tools
        self.error_inds = error_inds
        
    def binning(self, cols = None, pass_error = True, remain_tick = True, fit_mode = "b", **kwargs):
        """
        mode 拆分为 mode 和 fit_mode
        mode 决定分箱的类型 刻度 info的指标
        fit_mode 决定result选取哪些指标
        """
        binning_tools = self.binning_tools
        error_inds = self.error_inds        

        
        if fit_mode == "b":
            fit_func = lazy_fit_func
        if fit_mode == "c":
            fit_func = intc.fit_func

        #fit_func = lazy_fit_func

        if cols is None:
            cols = self.X.columns.tolist()
        kwargs["y"] = self.Y
        
        for i in cols:
            kwargs["x"] = self.X[i]
            try:
                if remain_tick and (i in binning_tools):
                    # if don't change ticks, apply calculate on each
                    binning_tools[i]. result = \
                        binning_tools[i]. calculate(
                            **{**model_flow.default_kwargs,
                               **kwargs,
                               "func": fit_func})
                else:
                    # if change ticks use binning.binning
                    binning_tools[i] = \
                        binning.binning(
                            **{**model_flow.default_kwargs,
                               **kwargs,
                               "func": fit_func})

            except Exception as e:
                if i in binning_tools:
                    del binning_tools[i]
                if pass_error:
                    self.error_inds.add(i)
                    continue
                else:
                    raise e

        self.binning_tools = binning_tools
        self.error_inds = error_inds

    def reset_info(self):
        """
        delete fit information
        but remain the boundary, left, right ...
        """
        binning_tools = self.binning_tools
        for i in binning_tools.values():
            i.reset_info()
        self.binning_tools = binning_tools

    
    def change_xy(self, x, y):
        _items = ["cmt",
                  "binning_tools",
        ]
        _kw = deepcopy({i: getattr(self, i) for i in _items})
        _t = model_flow().from_data(X = x, Y = y, **_kw)
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
        if cols is None:
            cols = self.binning_tools.keys()
        subs = self.subobj_condition(conds = conds, labels = labels)
        _ = [i.fit(cols = cols) for i in subs]
        _ = [i.binning(cols = cols) for i in subs]
        sub_binL = OrderedDict()
        sub_binning_tools = sub_binning()
        for i in subs:
            sub_binL[i.label] = i.binL()
            sub_binning_tools[i.label] = i.binning_tools

        self.sub_binL = sub_binL
        self.sub_binning_tools = sub_binning_tools
        self.sub_labels = labels

    def ent(self, i):
        return db_ent(self.binning_tools[i]. result[["bad_cnt", "good_cnt"]]. values)
        
    @property
    def entL(self):
        return pd.Series({i:self.ent(i) for i in self.binning_tools.keys()}, name = "ent").sort_values(ascending = False)

    def binL(self):
        _ent = self.entL
        _dfs = list()
        for i in _ent.index:
            _df = model_IO.add_porp_lift(self.binning_tools[i].\
                                    result.reset_name("区间").reset_name("number"))
            _df["指标"] = i
            _df["区分度"] = _ent.loc[i]
            _df["type"] = "int" if "intList" in str(type(self.binning_tools[i])) else "str"
            _dfs.append(_df)
        return pd.concat(_dfs)

    def var_find(self, draw = False):
        cs = index_selector(log = self)
        b = cs.var_find()
        b = pd.concat(b, axis = 1)
        b.columns = ["bif_mean", "bif_porp", "bif_ent"]
        
        self.index_bifurcate = b
        return b
    
    def update_woe(self, cols = None):
        if not hasattr(self, "woevalue"):
            self.woevalue = pd.DataFrame(index = self.X.index)
        if cols is None:
            cols = self.binning_tools.keys()
        for i in cols:
            self.woevalue[i] = self.trans(i, "woe").astype(float)
            
        self.corr = self.woevalue.corr()

    def load_index(cols):
        self.selected_index = cols

    def cluster(self, cols=None, ruleD=0.9, ruleN=3):
        if cols is None:
            cols = self.selected_index
            
        from cluster import Hierarchical
        DISTANCE = (1 - (self.corr.loc[cols, cols])**2)
        h = Hierarchical(data=DISTANCE)
        names = [(i, ) for i in cols]
        cluster_bundle = h.hcluster(col_list=names, ruleD=ruleD, ruleN=ruleN)
        cluster_bundle = [self.entL.loc[list(i)] for i in cluster_bundle]
        cluster_bundle.sort(key=lambda x:x.max(), reverse=True)
        
        self.cluster_bundle = cluster_bundle
        return cluster_bundle

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
        RES["trans"] = intLDict.load_format(RES["trans"])
        RES["trans_func"] = lambda x:RES["trans"]. trans(x[RES["cols"]], default = RES["standard_woe"]) - RES["standard_woe"]
        return RES

class model_IO:
    ###
    home = "/home/lishiyu/Project/model_vision"
    global_db = home + "/.model.db"
    
    def __init__(self):
        self.version_node = version_node()
        self.tmp_node = self.version_node
        self.lv_saved = [True, True, True, True]
        self.index_saved = defaultdict(bool)

    def version_dir(self, k):
        _v = self.tmp_node.v()
        _version = "/". join([str(int(i)) for i in _v[:(k + 1)]]) + "/"        
        return _version

    def save(self, i):
        j, k = mfs.static_variable[i]
        self.index_saved[i] = False
        self.lv_saved[k] = False        
        if j == "dir":
            return
        v = getattr(self.model, mfs.mf_v_prefix + i)
        if j in ["intLDict", "sub_binning"]:
            v = v.save_format()
        _dir = self.cache
        path = _dir + i + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(v, f)

    def load(self, i):
        j, k = mfs.static_variable[i]
        if j == "dir":
            return True
        path = self.cache + i + ".pkl"
        v = pd.read_pickle(path)
        if j in ["intLDict", "sub_binning"]:
            v = eval(j).load_format(v)
        return v
        
    def save_data(self, level):
        need_update = False

        for k in range(int(level) + 1):
            if self.lv_saved[k] is False:
                need_update = True
                low_level = k
                break

        if not need_update:
            return

        _v = self.tmp_node.v()
        _a = self.tmp_node.ancestor() + [self.tmp_node]
        low_level = min(len(_v), low_level)
        _p = _a[low_level]
        for k in range(low_level, level + 1):
            self.lv_saved[k] = True
            _p = _p.add_child()
        self.tmp_node = _p

        for i, (j, k) in mfs.static_variable.items():
            if i not in self.index_saved:
                continue
            if k > level:
                continue
            if k < low_level:
                continue
            self.save_stable(i)
            self.index_saved[i] = True

        pd.Series(self.version_node).to_pickle("tree_version.pkl")

    def save_stable(self, i):
        j, k = mfs.static_variable[i]
        if j == "dir":
            _loc = self.cache + i
        else:
            _loc = self.cache + i + ".pkl"
        _dir = self.dir + self.version_dir(k)
        os.system("mkdir -p {0}". format(_dir))
        os.system("cp {0} {1} -R". format(_loc, _dir))

    def from_model(self, model, cwd=None):
        if cwd is None:
            cwd = os.getcwd()
        self.cwd = cwd
        self.model = model
        self.model.tsp = "Lgt" + datetime.now().strftime("%Y_%m%d_%H%M%S_%s")        
        self.name = self.model.tsp
        self.model.model_IO = self
        self.dir = "{0}/.model_profile/{1}/".  format(self.cwd, self.name)
        self.common_dir = "{0}/.model_profile/".  format(self.cwd)
        self.cache = self.dir + ".cache/"
        self.b_png_dir = self.cache + "b_png/"
        self.b_html_dir = self.cache + "b_html/"
        self.saved_result = self.dir + "model_result/"        
        self.binning_excel_path = self.dir + "binning.xlsx"        
        self.has_record = False

        for _path in [
                self.common_dir,
                self.dir,
                self.cache,
                self.b_png_dir,
                self.b_html_dir,
                self.saved_result,
        ]:
            if not os.path.exists(_path):
                os.mkdir(_path)

        self.db = "{0}/.model_profile/{1}/model.db".  format(self.cwd, self.name)
        self.common_db = "{0}/.model_profile/model.db".  format(self.cwd)
        self.conn = sqlite3.connect(self.db)
        self.conn_common = sqlite3.connect(self.common_db)
        self.conn_global = sqlite3.connect(self.global_db)
        return self

    @staticmethod
    def add_porp_lift(_b):
        _b = _b.copy()
        _b["porp"] = _b["cnt"] / _b["cnt"]. sum()
        _b["lift"] = (_b["mean"] / (_b["cnt"] * _b["mean"]).sum() *\
                      _b["cnt"]. sum()).fillna(0)
        _b.r1({"cnt": "总数",
               "porp": "占比",
               "lift": "提升",
               "mean": "坏率",
        })
        return _b

    def save_binning_html_index(self, name, path, UPPER = 0.08):
        _bt = self.model.binning_tools[name]
        x_label = _bt.info["bins"]
        #name = "card_cm_bank12m_pct"
        info =  {l:s1[name].info for l, s1 in self.model.sub_binning_tools.items()}
        bar = echarts_plot(x_label, info, UPPER=UPPER)
        bar.chart_id = "{{id}}"
        bar.render(template_name = "simple_chart_body.html", path = path)

    def save_binning_html(self):
        for i in self.model.binning_tools.keys():
            path = self.b_html_dir + i + ".html"
            self.save_binning_html_index(i, path)

        self.model.b_html = True

    def add_name_cmt(self, name, sep = "&", l1 = 100, l2 = 100):
        _cmt = self.model.cmt.get(name, "")
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

    def save_binning_png(self, cols = None, draw_bad_rate = True, upper_lim = 0.06):
        if cols is None:
            cols = list(self.model.sub_binning_tools.values().__iter__().__next__().keys())
        for i in cols:
            _name = (self.add_name_cmt(i) + ".png").replace("/", "_")
            _title = self.add_name_cmt(i, sep = "\n", l1 = 30, l2 = 20)
            path = self.b_png_dir + _name
            _df = self.model.sub_binning_tools.sub_info(i)
            model_IO._barplot(_df,
                              _title, path,
                              draw_bad_rate = draw_bad_rate,
                              upper_lim = upper_lim)
            
        self.model.b_png = True
            
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
        res["trans"] = _d.save_format()
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
                    model_flow=None
    ):
        res = dict()
        res['model'] = model
        res['cols'] = cols
        res['ticks'] = ticks
        res['standard_woe'] = standard_woe
        res['trans'] = binning_tools.sub(cols).save_format()
        res['cond'] = train_cond
        self.result = res
        self.m = model_flow
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
        self.tools = intLDict.load_format(res["trans"])
        self.func = lambda x:self.tools.\
            trans(x[res["cols"]],
                  default = res["standard_woe"]) - res["standard_woe"]
        self.coef = pd.Series(
            res["model"].coef_[0],
            index = res["cols"],
            name = "系数")
        return self

    def load_format(self, res):
        self.result = res
        return self.load_func()

    def save_format(self):
        return self.result

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
        res["binning"] = flow_IO.add_porp_lift(_b)
        return res


    def binning_cond(self, cond):
        x = self.X[self.result["cols"]].loc[cond]
        y = self.Y.loc[cond]
        return self.binning_sample(x, y)




class version_node(list):
    def __init__(self):
        self.parent = None
        
    def add_child(self):
        p = version_node()
        self.append(p)
        p.parent = self
        return p

    def number(self):
        for j, i in enumerate(self.parent):
            if i is self:
                return j

    def ancestor(self):
        if self.parent is None:
            return []
        else:
            return self.parent.ancestor() + [self.parent]

    def v(self):
        return [i.number() for i in (self.ancestor() + [self])[1:]]

    def save_format(self):
        if len(self) == 0:
            return []
        else:
            return [i.save_format() for i in self]

    @staticmethod
    def load_format(v):
        p = version_node()
        if len(v) == 0:
            return p
        for i, _v in enumerate(v):
            _p = version_node.load_format(_v)
            p.append(_p)
            _p.parent = p
        return p
    

class mfs(model_flow):
    mf_v_prefix = "_mf_variable_"

    #static_variable = {"a0": ("raw", 0), 
    #                   "b0": ("raw", 0),
    #                   "a1": ("raw", 1),
    #                   "b1": ("raw", 1),
    #                   "a2": ("raw", 2),
    #                   "b2": ("raw", 2),
    #                   
    #}
    
    static_variable = {
        "binning_tools": ("intLDict", 0), 
        "sub_binL": ("raw", 0), 
        "sub_binning_tools": ("sub_binning", 0), 
        "sub_labels": ("raw", 0),
        "corr": ("raw", 0),
        "index_bifurcate": ("raw", 0),
        "b_html": ("dir", 0),
        "b_png": ("dir", 0),
        
        "selected_index": ("raw", 1),
        "cluster_bundle": ("raw", 2),
    }
    
    def from_data(self, *args, **kwargs):
        self.model_IO = model_IO().from_model(self)
        return super().from_data(*args, **kwargs)
    
    def getv(self, i):
        try:
            if hasattr(self, self.mf_v_prefix + i):
                return getattr(self, self.mf_v_prefix + i)
            else:
                return getattr(self, "load_" + i)()
        except:
            return None
        
    def setv(self, v, i):
        setattr(self, self.mf_v_prefix + i, v)
        getattr(self, "save_" + i)()
        return True

    def save(self, i):
        return getattr(self.model_IO, "save_" + i)()

    def load(self, i):
        return getattr(self.model_IO, "load_" + i)()

    save_binning = lambda self:self.model_IO.save_data(level=0)
    save_index = lambda self:self.model_IO.save_data(level=1)    
    save_cluster = lambda self:self.model_IO.save_data(level=2)
    save_result = lambda self:self.model_IO.save_data(level=3)
    
for i, (j, k) in mfs.static_variable.items():
    setattr(mfs, "_getv_" + i, partialmethod(mfs.getv, i=i))
    setattr(mfs, "_setv_" + i, partialmethod(mfs.setv, i=i))
    setattr(mfs, i, property(getattr(mfs, "_getv_" + i)))
    setattr(mfs, i, getattr(mfs, i).setter(getattr(mfs, "_setv_" + i)))
    
    setattr(mfs, "save_" + i, partialmethod(mfs.save, i=i))
    setattr(mfs, "load_" + i, partialmethod(mfs.load, i=i))
    setattr(model_IO, "save_" + i, partialmethod(model_IO.save, i=i))
    setattr(model_IO, "load_" + i, partialmethod(model_IO.load, i=i))

if __name__ == "__test__":
    class A():
        def getv(self, j):
            return self._i + j

        def setv(self, v):
            self._i = v

        def getvs(self, i, j):
            return self._i + j

    A.i = partialmethod(A.getv, j=1)
    A.i = property(A.i)
    A.i = A.i.setter(A.setv)

    sgg = pd.DataFrame([{"data":pickle.dumps({"a": 1}), 
                         "asd": 1, 
    }])
    
