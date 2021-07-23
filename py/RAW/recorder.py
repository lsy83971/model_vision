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
from RAW.int import binList, intLDict

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

class recorder:
    home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    global_db = home + "/.model.db"
    def from_m(self, m, cwd=None):
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
        self.cache = self.dir + ".cache/"
        self.saved_result = self.dir + "model_result/"        
        self.binning_excel_path = self.dir + "binning.xlsx"        
        self.has_record = False

        for _path in [
                self.common_dir,
                self.dir,
                self.b_png_dir,
                self.b_html_dir,
                self.b_tool_dir,
                self.cache,
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

    def from_name(self, name, db, dir):
        self.name = name
        self.db = db
        self.dir = dir
        self.b_png_dir = self.dir + "b_png/"
        self.b_html_dir = self.dir + "b_html/"
        self.cache = self.dir + ".cache/"
        self.binning_excel_path = self.dir + "binning.xlsx"
        self.saved_result = self.dir + "model_result/"                
        self.common_dir = os.path.dirname(self.dir[: -1]) + "/"
        self.common_db = self.common_dir + "model.db"
        import sqlite3
        self.conn = sqlite3.connect('{0}'.  format(self.db))
        self.conn_common = sqlite3.connect(self.common_db)
        self.conn_global = sqlite3.connect(self.global_db)
        return self

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

    def r_t_cols(self):
        self.save_df(self.m.X.columns.tolist(), "t_columns.pkl")
    def load_t_cols(self):
        return self.load_df("t_columns.pkl")
    
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

    def load_binning_tools(self):
        _d = self.load_df("binning_tools.pkl")
        return intLDict.read_pattern(_d)
    
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
        return self.load_table("comment").set_index("index")["comment"]
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
        info =  {l:s1[name].info for l, s1 in self.m.sub_binning_tools.items()}
        bar = echarts_plot(x_label, info, UPPER=UPPER)
        bar.chart_id = "{{id}}"
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
        self.save_table(pd.concat([self.m.X, self.m.Y], axis = 1).reset_name("index"), "xy")

    def r_woe(self):
        self.save_table(pd.concat([self.m.X["dt"], self.m.woevalue, self.m.Y], axis = 1).reset_name("index"), "woe")
        
    def load_x(self, cols=None):
        if cols is None:
            cols1 = '*'
        else:
            cols1 = ",". join([i for i in cols])
        return pd.read_sql("select {1} from {0} order by `index`".  format(self.add_name("xy"), cols1), self.conn)

    def load_y(self):
        return pd.read_sql("select label from {0} order by `index`".  format(self.add_name("xy"), cols1), self.conn)

    def load_woe_x(self, cols=None):
        if cols is None:
            cols1 = '*'
        else:
            cols1 = ",". join([i for i in cols])
        return pd.read_sql("select {1} from {0} order by `index`".  format(self.add_name("woe"), cols1), self.conn)

    def load_woe_y(self):
        return pd.read_sql("select label from {0} order by `index`".  format(self.add_name("woe"), cols1), self.conn)

    
    
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
        
    def save_cluster(self, cols, symbol = "test"):
        _df = pd.DataFrame([json.dumps(cols)])
        _df.columns = ["cols"]
        _df["dt"] = str(datetime.now())
        _df["symbol"] = symbol
        self.save_table(_df, "cluster", append = True)

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
            worksheet.write(2 * _i1, 2 * _i2 + 1,"{0}".format(cols[i]))
            DRAW.draw.insert_image(worksheet,
                                   _d[cols[i]],
                                   row = 2 * _i1 + 1,
                                   col = 2 * _i2 + 1,
                                   x_scale = 3 ,
                                   y_scale = 2)
        workbook.close()

    def r_binning_excel(self, cols = None):

        self._excel_addpng(path = self.binning_excel_path, cols = cols)
        with pd.ExcelWriter(self.binning_excel_path, mode = "a", engine = "openpyxl") as writer:
            self.m.binL(). to_excel(writer, sheet_name = "全量",index = False)
            for i, j in self.m.sub_binL.items():
                j. to_excel(writer, sheet_name = i, index = False)

    def load_binl(self):
        return pd.read_excel(self.binning_excel_path, sheet_name="全量")

    def r_model_sample(self):
        path = self.dir + "modelsample.csv"
        self.sample_path = path
        self.m.oX[self.m.model_cols]. sample(min(1000, self.m.oX.shape[0])). to_csv(path, index = False)

    def r_model_result(self):
        path = self.dir + "modelresult.pkl"
        self.model_result_path = path
        self.m.model_result.save(path)

    def r_model_report(self, mr=None, path=None):
        try:
            m = self.m
            mr = m.model_result
            _coef = mr.coef
            binl = m.binL()
            entl = m.entL
            cmt = m.cmt
            cols = m.X.columns
            standard_woe = m.standard_woe
            train_cond = m.train_cond[i]
            X = m.X
        except:
            assert mr is not None
            _coef = mr.coef
            binl = self.load_binl()
            entl = self.load_ent()
            cmt = self.load_comment()
            cols = self.load_t_cols()
            #mr = sb.model_result
            _cols = list(set(mr.result["cols"] + ["label", "dt", "month", "channel"]) & set(cols))
            _cols1 = mr.result["cols"] + ["label"]
            _cols2 = list(set(cols) & {"channel", "month", "dt"})
            woe_xy = self.load_woe_x(_cols1)
            woe_x = woe_xy.drop("label", axis=1)[mr.result["cols"]]
            xy = self.load_x(_cols2)            
            X = xy
            y = woe_xy["label"]
            standard_woe = math.log(((y == 1).sum() + 0.5) / ((y == 0).sum() + 0.5))
            train_cond = mr.result["cond"]
            assert train_cond is not None
            mr = mr.load_xy(woe_x, y)

        if path is None:
            path = self.dir + "modelreport.xlsx"
        self.model_report_path = path

        try:
            self._excel_addpng(path, cols = _coef.index.tolist(),
                               sheet_name = "2.2分箱图")
        except:
            pass
        single_binning = binl.merge(cmt, left_on = "指标", right_index = True)
        single_binning["woe"] = single_binning["woe"] - standard_woe
        coef = pd.concat([_coef, cmt, entl], axis = 1, join = "inner").reset_name("指标名")
        print(train_cond.keys())
        print(train_cond)
        b_sample = [mr.binning_cond(train_cond[i])["binning"]. reset_name(i) for i in sorted(train_cond)]
        result = dict()
        result["2.1单指标分箱"] = [single_binning]
        result["3.1模型参数"] = [coef]
        result["4.1样本分箱"] = b_sample

        for i in ["month", "channel"]:
            if i not in cols:
                continue
            _v = list(set(X[i]))
            _v.sort()
            _l1 = list()
            for k in _v:
                _cond = (X[i] == k)
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

    def r_ent(self):
        self.save_df(self.m.entL, "ent.pkl")
    def load_ent(self):
        return self.load_df("ent.pkl")
    def load_cluster(self):
        corr = self.load_corr()
        ent = self.load_ent()
        return col_cluster.from_data(data=corr, entL=ent)

class loader:
    home = recorder.home
    global_db = recorder.global_db

    def __init__(self, name, db, dir):
        self.name = name
        self.db = db
        self.dir = dir
        self.b_png_dir = self.dir + "b_png/"
        self.b_html_dir = self.dir + "b_html/"
        self.cache = self.dir + ".cache/"
        
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
    save_cols = recorder.save_cols
    save_cluster = recorder.save_cluster
    load_binning_tools = recorder.load_binning_tools
    load_sub_binning_tools = recorder.load_sub_binning_tools
    load_sub_cond = recorder.load_sub_cond
    load_recorder = recorder.load_recorder

    load_html_path = recorder.load_html_path
    load_html_files = recorder.load_html_files
    load_html_map = recorder.load_html_map


    load_bifurcate = recorder.load_bifurcate
    load_cols = recorder.load_cols
    load_comment = recorder.load_comment
    load_corr = recorder.load_corr
    load_ent = recorder.load_ent
    load_cluster = recorder.load_cluster
    load_x = recorder.load_x
    load_y = recorder.load_y
    load_woe_x = recorder.load_woe_x
    load_woe_y = recorder.load_woe_y


