import math
import os, sys
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import sys
from RAW.int import *
from RAW.ent import db_ent
from default_var import *
import xlsxwriter

try:
    from default_var import *
    add_guests("logit_user_cnt.txt")
except:
    pass

def cond_part(_dt, _l):
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

    def __init__(self, X, Y, cmt = None, **kwargs):
        add_guests("lgt_init.txt")
        self.raw_X = X
        self.raw_Y = Y
        self.X = X.reset_index(drop = True).replace("", -999999).fillna(-1)
        dtypes = self.X.dtypes.astype(str)
        _trans_type_dict = dict()
        for i in dtypes.index:
            t = dtypes[i]
            if t != "object":
                continue
            i1, i2 = binning.guess_type(self.X[i])
            if i1 == "int":
                print("as float", i)
                _trans_type_dict[i] = i2

        if len(_trans_type_dict) > 0:
            _X2 = pd.DataFrame(_trans_type_dict)
            self.X.drop(list(_trans_type_dict.keys()), axis = 1, inplace = True)
            self.X = pd.concat([self.X, _X2], axis = 1)

        self.Y = Y.reset_index(drop = True)
        self.Y.name = "label"
        if cmt is None:
            cmt = pd.Series(X.columns, index = X.columns, name = "comment")
        else:
            cmt = cmt. cv(self.X.columns)
        self.cmt = cmt
        self.cmt.name = "注释"
        for i, j in kwargs.items():
            setattr(self, i, j)
        self.init_bintool()
        self.init_tsp()
        self.init_basic()

    def binning_cnt(self):
        return pd.Series({i: len(j) for i, j in self.binning_tools.items()})

    def init_basic(self):
        self.standard_woe = math.log(((self.Y == 1).sum() + 0.5) / ((self.Y == 0).sum() + 0.5))

    def init_tsp(self):
        self.tsp = "Lgt" + datetime.now().strftime("%Y%m%d_%H%M")

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
        _t = lgt(X = x, Y = y, **_kw)
        _t.reset_info()
        _t.fit()
        return _t

    def change_xy_cond(self, cond):
        return self.change_xy(x = self.X.loc[cond], y = self.Y.loc[cond])

    def subobj_condition(self, conds, labels):
        _l = list()
        for i in range(len(conds)):
            _o = self.change_xy_cond(conds[i])
            _o.label = labels[i]
            _o.part = str(int(i + 1))
            _l.append(_o)
        return _l

    def init_png_dir(self):
        if not hasattr(self, "png_dir"):
            self.png_dir = "./" + self.tsp + "_png/"
        if not os.path.exists(self.png_dir):
            os.mkdir(self.png_dir)
        self.binning_excel_path = './' + self.tsp + ".xlsx"
        self.png_dict = dict()

    def draw_binning(self,
                     conds,
                     cols = None,
                     labels = None,
                     draw_bad_rate = True,
                     upper_lim = None
                    ):
        if not hasattr(self, "png_dict"):
            self.init_png_dir()
        if labels is None:
            labels = [str(i + 1) for i in range(len(conds))]
        self.binning_labels = labels
        if cols is None:
            cols = self.binning_tools.keys()
        subs = self.subobj_condition(conds = conds, labels = labels)
        _ = [i.fit(cols = cols) for i in subs]
        _ = [i.binning(cols = cols) for i in subs]

        for i in cols:
            i1 = i
            i2 = self.cmt.get(i, "")
            path = self.png_dir + (i1 + "&" + i2 + ".png").replace("/", "_")
            self.png_dict[i] = path
            if len(i1) > 30:
                i1 = i1[:30] + "..."
            if len(i2) > 20:
                i2 = i2[:20] + "..."
            title = i1 + "\n" + i2
            self.binning_barplot(subs, i, title, path, draw_bad_rate = draw_bad_rate, upper_lim = upper_lim)

        self.sub_binL = {i.label:i.binL() for i in subs}
        self.sub_binning_tools = {i.label:i.binning_tools for i in subs}
        return subs

    def draw_excel_addpng(self, cols, worksheet):
        import DRAW.draw
        for i in range(len(cols)):
            _i1, _i2 = int(i / 3),i % 3
            worksheet.write(2 * _i1, 2 * _i2 + 1,"{0}".format(cols[i]))
            DRAW.draw.insert_image(worksheet,
                                   self.png_dict[cols[i]],
                                   row = 2 * _i1 + 1,
                                   col = 2 * _i2 + 1,
                                   x_scale = 3 ,
                                   y_scale = 2)

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

    def draw_binning_excel_addpng(self, path, cols = None, sheet_name = "分箱图"):
        if cols is None:
            cols = list(self.png_dict.keys())
        _ent = self.entL
        cols.sort(key = lambda x:_ent.get(x.split("&")[0], 0), reverse = True)
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet(sheet_name)
        self.draw_excel_addpng(cols, worksheet)
        workbook.close()

    def draw_binning_excel(self, cols = None):
        self.draw_binning_excel_addpng(path = self.binning_excel_path, cols = cols)
        with pd.ExcelWriter(self.binning_excel_path, mode = "a", engine = "openpyxl") as writer:
            self.binL(). to_excel(writer, sheet_name = "全量",index = False)
            for i, j in self.sub_binL.items():
                j. to_excel(writer, sheet_name = i, index = False)

    @staticmethod
    def binning_barplot_df(subs, i):
        _dfs = list()
        for k in range(len(subs)):
            _df = subs[k]. binning_tools[i]. result. copy()
            _df["label"] = subs[k]. label
            _df["part"] = subs[k]. part
            _df["porp"] = _df["cnt"] / _df["cnt"]. sum()
            _df["bad_rate"] = _df["mean"]
            _df.reset_name("bin", inplace = True)
            _df.reset_name("code", inplace = True)
            _dfs.append(_df)
        t_df = pd.concat(_dfs)
        return t_df

    @staticmethod
    def binning_barplot(subs, i, title, path, draw_bad_rate = True, upper_lim = None):
        import DRAW.draw
        t_df = lgt.binning_barplot_df(subs, i)
        t_df_copy = t_df.copy()
        if upper_lim is not None:
            t_df_copy["bad_rate"] = t_df_copy["bad_rate"]. apply(lambda x:min(x, upper_lim))
        DRAW.draw.draw_bar(t_df_copy,
                           title = title,
                           save = path,
                           draw_bad_rate = draw_bad_rate)

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
            _df = lgt.add_porp_lift(self.binning_tools[i].\
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
                _t = binning.tick(x = _score, quant = quant, single_tick = False, ruleV = _score.shape[0] / quant / 2, mode = "v").ticks_boundaries()
                self.score_ticks = _t
                RES = self.save_model_result()

                _res1 = dict()
                for i in range(len(sample)):
                    _res1[labels[i]] = self.sample_binning(sample[i])
                RES["result"] = _res1
                self.model_result = RES
                return RES

    def sample_binning(self, cond):
        res = dict()
        cols = self.model_cols
        _x1 = self.woevalue.loc[cond, cols]
        _y1 = self.Y.loc[cond]
        _score = pd.Series(self.model.predict_proba(_x1)[:, 1], index = _x1.index)
        res["KS"] = KS(_y1, _score)
        res["AUC"] = roc_auc_score(_y1, _score)
        z = pd.concat([_y1, _score], axis = 1)
        z.columns = ["label", "x"]
        _b = z.binning(x = "x", y = "label", l = self.score_ticks)
        res["binning"] = lgt.add_porp_lift(_b)
        return res

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

    def predict(self, X):
        X1 = self.trans_woe(cols = self.model_cols, X = X)[self.model_cols]
        return self.predict1(X1)

    def predict1(self, X1):
        X1 = X1[self.model_cols]
        return pd.Series(self.model.predict_proba(X1)[:, 1], index = self.X.index, name = "score")

    def save_model_sample(self):
        path = self.tsp + "_modelsample.csv"
        self.model_sample_path = path
        self.raw_X[self.model_cols]. sample(min(1000, self.raw_X.shape[0])). to_csv(path, index = False)

    def save_model_result(self):
        path = self.tsp + "_modelresult.pkl"
        self.model_result_path = path
        RES = dict()
        RES["model"] = self.model
        RES["cols"] = self.model_cols
        _d = intLDict({i: j for i, j in self.binning_tools.items() if i in self.model_cols})
        _d.cover_inf()
        RES["trans"] = _d.write_pattern()
        RES["ticks"] = self.score_ticks
        RES["standard_woe"] = self.standard_woe
        with open(path, "wb") as f:
            pickle.dump(RES, f)
        return RES

    def save_model_report(self):
        path = self.tsp + "_model_report.xlsx"
        self.model_report_path = path
        result = dict()
        _coef = pd.Series(self.model.coef_[0], index = self.model_cols, name = "系数")
        result["2.1单指标分箱"] = [self.binL().merge(self.cmt, left_on = "指标", right_index = True)]
        result["2.1单指标分箱"][0]["woe"] = result["2.1单指标分箱"][0]["woe"] - self.standard_woe
        result["3.1模型参数"] = [pd.concat([_coef, self.cmt, self.entL],
                                           axis = 1, join = "inner").\
                                 reset_name("指标名"),
                                 #pd.DataFrame(pd.Series(self.model.intercept_,name="value",
                                 #               index = ["Intercept"])).\
                                 #reset_name("Intercept"),
                                 ]
        if hasattr(self, "png_dir"):
            self.draw_binning_excel_addpng(path = path, cols = _coef.index.tolist())

        for i in ["month", "channel"]:

            if i == "month":
                i1 = "4.2月份分箱"
            if i == "channel":
                i1 = "5.1渠道分箱"
            _l1 = list()
            if i not in self.X.columns:
                continue
            _v = list(set(self.X[i]))
            _v.sort()
            for k in _v:
                _cond = (self.X[i] == k)
                _res = self.sample_binning(_cond)
                _l1.append(_res["binning"].reset_name(k))
            result[i1] = _l1

        result["4.1样本分箱"] = \
        [self.model_result["result"][i]["binning"]. reset_name(i) for i in sorted(self.model_result["result"]. keys())]
        from default_var import excel_sheets
        for i in sorted(result.keys()):
            excel_sheets(path, result[i], i)

    def model_online(self, model_name, channel, container):
        """
        kw["model_name"] = "自营天衍模型V1"
        kw["channel"] = "ZY_TY_V1"
        kw["container"] = "model_image1"
        """
        sample_path = self.tsp + "_modelsample.csv"
        result_path = self.tsp + "_modelresult.pkl"
        report_path = self.tsp + "_model_report.xlsx"

        kw = dict()
        kw["model_name"] = model_name
        kw["channel"] = channel
        kw["container"] = container
        kw["date"] = datetime.now().strftime("%Y%m%d")
        kw["model_cols"] = "\n". join([str(i + 1) + ". " + j for i, j in enumerate(self.model_cols)])
        kw["model_cols_json"] = self.raw_X.iloc[0][self.model_cols]. to_dict()


        os.system("cp {0} {1}.csv". format(sample_path, kw["channel"]))
        os.system("cp {0} {1}.pkl". format(result_path, kw["channel"]))
        os.system("cp {0} {1}模型报告.xlsx". format(report_path, kw["channel"]))
        with open("/home/bozb/notebook/lsy/PYLIB/MODEL/FILE/online_file.txt", "r") as f:
            fs = f.read()

        with open("{0}_{1}.txt". format(kw["model_name"], kw["channel"]), "w") as f:
            f.write(fs.format(**kw))

    @staticmethod
    def load_model_result(path):
        RES = pd.read_pickle(path)
        RES["trans"] = intLDict.read_pattern(RES["trans"])
        RES["trans_func"] = lambda x:RES["trans"]. trans(x[RES["cols"]], default = RES["standard_woe"]) - RES["standard_woe"]
        return RES

    @staticmethod
    def save_model_result1(RES, path):
        del RES["trans_func"]
        RES["trans"] = RES["trans"].write_pattern()
        with open(path, "wb") as f:
            pickle.dump(RES, f)
        return RES

    def var_find(self, path = None, draw = False):
        if path is None:
            path = self.binning_excel_path
        be = binning_excel(log = self)
        b = dict()
        b[0] = pd.Series({i: be.mean_dif_ent(i) \
                        for i in be.total.keys()}).sort_values(ascending = False)
        b[1] = pd.Series({i: be.porp_dif_ent(i) \
                        for i in be.total.keys()}).sort_values(ascending = False)
        b[2] = self.entL.sort_values(ascending = True)
        self.index_bifurcate = b
        if draw:
            for i in range(2):
                _b = b[i].index.tolist()
                _b_filename = self.tsp + "_b{0}.xlsx". format(str(i))
                workbook = xlsxwriter.Workbook(_b_filename)
                worksheet = workbook.add_worksheet("compare")
                self.draw_excel_addpng(cols = _b, worksheet = worksheet)
                workbook.close()
        return b

class binning_excel:
    def __init__(self, path = None, log = None):
        assert not(path is None and log is None)
        import xlrd
        if path is None:
            path = log.binning_excel_path
        self.path = path
        data = xlrd.open_workbook(path) #打开demp.xlsx文件
        sheet_names = data.sheet_names()  #获取所有sheets名
        self.total_label = sheet_names[1]
        self.labels = sheet_names[2:]
        info = dict()
        for j, i in enumerate(sheet_names[1:]):
            d = pd.read_excel(path, sheet_name = i)
            d1 = \
            {j1:j2.set_index("number") for j1, j2 in d.groupby("指标", as_index = False)}
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
        from RAW.db import db_ent
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
            if bt.loc[i, "is_stick"]:
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
        _res_dif = _res.loc[(_res["is_stick"]) == False]["坏率"]. diff().iloc[1:]
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
            print("G")
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

def var_find(b_file, sheets):
    t_ent = pd.read_excel(b_file,sheet_name = "全量").drop_duplicates("指标").set_index(["指标"])["区分度"]
    t_df1 = pd.read_excel(b_file,sheet_name = "全量")
    s_df = [pd.read_excel(b_file,sheet_name=i).set_index(["指标","区间"]) for i in sheets]
    for j, i in enumerate(s_df):
        i["rate"] = i["坏率"]
        i["label"] = j
        i["bad"] = i["总数"] * i["rate"]
        i["good"] = i["总数"] - i["bad"]

    init_cnt = pd.Series(1, index = s_df[0]. index, name = "cnt")
    for i in s_df:
        init_cnt *= i["总数"]
    init_cnt_avg = init_cnt ** (1 / len(s_df))

    virtual_dfs = [pd.DataFrame(init_cnt_avg.copy()) for i in s_df]
    for j, i in enumerate(virtual_dfs):
        i["rate"] = s_df[j]["坏率"]
        i["label"] = j
        i["bad"] = i["cnt"] * i["rate"]
        i["good"] = i["cnt"] - i["bad"]

    virtual_info_t = pd.concat(virtual_dfs).reset_index()
    virtual_ent_t = virtual_info_t.groupby(["指标", "label"]).agg({"bad": "sum", "good": "sum"}).groupby("指标").apply(lambda x:db_ent(x[["good", "bad"]]. values))
    virtual_ent_1 = virtual_info_t.groupby(["指标", "区间"]).\
        apply(lambda x:pd.Series(
            {"ent": db_ent(x[["good", "bad"]]. values),
             "cnt": x[["good", "bad"]]. sum().sum()
            }))
    virtual_ent_1["ent_weight"] = virtual_ent_1["cnt"] * virtual_ent_1["ent"]
    virtual_ent_2 = virtual_ent_1.groupby("指标").agg({"cnt": "sum", "ent_weight": "sum"})
    virtual_ent_2["ent"] = virtual_ent_2["ent_weight"] / virtual_ent_2["cnt"]
    b0 = (virtual_ent_2["ent"] - virtual_ent_t).sort_values()


    b1 = pd.concat(s_df).reset_index()[["指标", "label", "区间", "总数"]]. \
        groupby(["指标"]).apply(lambda x:db_ent(x.set_index(["label", "区间"])["总数"].\
                                           unstack("区间").values)).sort_values()

    b2 = t_df1.drop_duplicates("指标").set_index("指标")["区分度"]. sort_values(ascending = False)
    return b0, b1, b2

def var_find1(b_file, sheets):
    t_ent = pd.read_excel(b_file,sheet_name = "全量").drop_duplicates("指标").set_index(["指标"])["区分度"]
    t_df1 = pd.read_excel(b_file,sheet_name = "全量")
    s_df = [pd.read_excel(b_file,sheet_name=i).set_index(["指标","区间"]).reset_index() for i in sheets]

    sum_bads = list()
    sum_goods = list()
    for i in s_df:
        i["bad"] = i["总数"]*i["坏率"]
        i["good"] = i["总数"]*(1-i["坏率"])

    sum_bads = [i[i["指标"] == i.iloc[0]["指标"]]["bad"]. sum() for i in s_df]
    sum_goods = [i[i["指标"] == i.iloc[0]["指标"]]["good"]. sum() for i in s_df]
    t_rate = sum(sum_bads) / sum(sum_goods)

    for j, i in enumerate(s_df):
        i["bad"] = i["bad"] * (sum_goods[j]) / (sum_bads[j]) * t_rate
        i["type"] = sheets[j]

    b_total = pd.concat(s_df)
    sub_ent = b_total[["type","指标","区间","good","bad"]].groupby(["指标","区间"]).apply(lambda x:db_ent(x[["good","bad"]].values))
    sub_ent1 = b_total[["type","指标","区间","good","bad"]].groupby(["指标","区间"]).apply(lambda x:x[["good","bad"]]. sum().sum()).reset_index()
    sub_ent2 = sub_ent1.set_index(["指标", "区间"]).groupby("指标").apply(lambda x:x / x.sum())
    sum_ent = (sub_ent * sub_ent2[0]).groupby("指标").agg({0:"sum"})
    sum_ent1 = b_total.groupby(["指标","type"]).agg({"good":"sum","bad":"sum"}).groupby("指标").apply(lambda x:db_ent(x.values))
    dif_ent_gb = (sum_ent[0]-sum_ent1).sort_values()
    dif_ent_gb = (dif_ent_gb / (b_total.drop_duplicates(["指标", "区间"]).groupby("指标").apply(lambda x:x.shape[0]) + 0.001).apply(lambda x:math.log(x))).sort_values()
    dif_ent_cnt = b_total.set_index(["指标","type","区间"])["总数"].unstack("type").groupby("指标").apply(lambda x:db_ent(x.values)).sort_values()
    span_ind = t_df1[t_df1["区间"]. str[0]. isin(["(", "["])]["指标"]. drop_duplicates().values
    return dif_ent_gb[dif_ent_gb.index.isin(span_ind)], dif_ent_cnt[dif_ent_cnt.index.isin(span_ind)], t_ent




if __name__ == "__main__":
    from VRB.save_pboc2_concat import concat
    x_raw = concat("x_XSMY_2020-12-06_2020-12-11.pkl")
    x_raw["label"] = (x_raw["derived_pbc_sum_l6m_cc_avg_amt_and_l24m_asfbal_pl"]. replace("", -1).astype(float)) > 100000
    from default_var import read_cmt
    log2 = lgt(X = x_raw.iloc[:, :30], Y = x_raw["label"], cmt = read_cmt())
    log2.binning(pass_error = True)
    log2.X["month"] = log2.X["pid"]. str[:11]
    cond1 = log2.X["pid"] > "PID20201208"
    cond2 = log2.X["pid"] <= "PID20201208"
    conds = [cond1, cond2]
    log2.draw_binning(conds = conds, cols = ["loan_credit_unpaid_cnt"])
    log2.update_woe()
    log2.woevalue["month"]
    cols = log2.var().index.tolist()
    log2.binning_tools["month"]
    res = log2.train(cols = cols, sample = conds)
    log2.save_model_report()
