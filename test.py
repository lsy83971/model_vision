## sudo pip install numpy pandas lightgbm flask matplotlib seaborn scikit-learn xlsxwriter xlrd==1.2.0 openpyxl
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Bar, Line, Grid
import pandas as pd


sample1 = pd.read_csv("./py/sample1.csv")
y = pd.read_csv("./py/y.csv")["odhis30_third"]
sample1["dt"] = pd.Series(pd.date_range("2020-01-01", "2021-01-01")).sample(sample1.shape[0], replace = True).values
import sys
sys.path.append("./py")

from importlib import reload
import RAW.logitSDK
reload(RAW.logitSDK)
reload(RAW.cluster)

from RAW.logitSDK import lgt, intLDict, cond_part, loader
sb = lgt(X = sample1, Y = y)
#sb = lgt(X = sample1.astype(str), Y = y)
sb.binning(mode = "B", ruleB = 7)
conds = cond_part(pd.to_datetime(sb.X["dt"]), 0.5)
sb.draw_binning(conds = conds)


sb.recorder.r_binning_png()
sb.recorder.r_binning_excel()

sb.update_woe()
ss = sb.var_find()

cols = sb.var().index.tolist()
cols = [i for i in cols if i != "month"]
train_cond = cond_part(sb.X["dt"], 0.7)
res = sb.train(cols = cols, sample = train_cond)

sb.recorder.add_record()
sb.recorder.r_binning_html()
sb.recorder.r_sub_cond()
sb.recorder.r_xy()
sb.recorder.r_woe()
sb.recorder.r_cmt()
sb.recorder.r_binning_tools()
sb.recorder.r_sub_binning_tools()
sb.recorder.r_bifurcate()
sb.recorder.r_corr()
sb.recorder.r_model_result()
sb.recorder.r_model_sample()
sb.recorder.r_ent()
sb.recorder.r_t_cols()
sb.recorder.r_model_report(mr=sb.model_result)


sb.recorder.load_woe_x(["month"])

from flask import render_template
#render_template("/home/lishiyu/Project/model_vision/.model_profile/Lgt2021_0722_1435_1626935739/b_html/card_cm_bank12m_pct.html", id="asd")

cl = sb.recorder.load_cluster()
cl.cluster(["month", "card_cm_bank12m_pct"])
cl.cluster(["month", "card_cm_bank60m_pct", "card_cm_bank12m_pct"], ruleD=0.9)[0]
cl.cluster(["month", "card_cm_bank60m_pct", "card_cm_bank12m_pct"], ruleD=0.9)[1]

sb.recorder.name
import flask

    
ss = sb.recorder.load_cluster()
ent = sb.recorder.load_ent()
corr = sb.recorder.load_corr()
a = col_cluster.from_data(data=corr, entL=ent)
a.cluster(corr.index.tolist())[0]
a.cluster(corr.index.tolist())[1]
a.cluster(corr.index.tolist())[2]
a.cluster(corr.index.tolist())

sb.recorder.load_sub_cond()["1"]["cond"]
sb.recorder.load_bifurcate()

sb.recorder.load_table("sub_binning")
sb.recorder.load_table("binning")
sb.recorder.load_table("html_path")
sb.recorder.load_table("bifurcate")
sb.recorder.load_corr()
pd.read_sql("select * from `Lgt2021_0714_1714_1626254097.comment`", sb.recorder.conn)
sb.recorder.load_cmt()

sb.recorder.conn
#pd.read_sql("drop table records", sb.recorder.conn1)

sb.recorder.r_sub_binning_tools()
sb.recorder.r_cmt()
sss = sb.recorder.load_table("comment")
pd.read_sql("select * from `Lgt2021_0713_1100_1626145234.comment`", sb.recorder.conn)

sb.cmt.reset_index().to_sql("`Lgt2021_0713_1100_1626145234.comment`", sb.recorder.conn)
sb.recorder. save_table(sb.cmt.reset_index(), "comment")
sb.recorder.db
sb.recorder.name

gg = loader(db = sb.recorder.db, dir = sb.recorder.dir)
sss = gg.load_sub_binning_tools()

gg = pd.read_sql("select * from records", sb.recorder.conn1)
sb.recorder.r_binning_html()
sb.sub_binning_tools.keys()
sb.sub_binning_tools["1"]["card_cm_bank12m_pct"].  info
sb.sub_binning_tools["2"]["card_cm_bank24m_pct"].  info


import sqlite3
conn = sqlite3.connect('./.model_profile/test.db')

_df.to_sql("{0}.asdf".  format(sb.tsp), conn, index = False)
_df.to_sql("asd.asdfg", conn, index = False)
pd.read_sql("select * from `{0}.asdf`".  format(sb.tsp), conn)

import copy
sbb = copy.deepcopy(sb.binning_tools)
for i in sbb.values():
    i.reset_info()

import copy
lgt_dict = dict()
labels = ["2021-03", "2021-04"]
for label in labels:
    _cond = np.random.random(sample1.shape[0]) < 0.1
    _sbb = copy.deepcopy(sbb)
    for j, i in _sbb.items():
        i.fit(x = sb.X[j].loc[_cond], y = y.loc[_cond])
    lgt_dict[label] = copy.deepcopy(_sbb)

sg["1"]["card_cm_bank12m_pct"].  label
sb.sub_binning_tools["1"]["card_cm_bank12m_pct"]
sb.sub_binning_tools["1"].  label

sg = sub_binning(sb.sub_binning_tools)
sg.write("a.pkl")
sgg = sub_binning.read("a.pkl")


self = sb.recorder
name = "card_cm_bank12m_pct"


def binning_html(self, name, path, UPPER = 0.08):
    _bt = self.m.binning_tools[name]
    x_label = _bt.info["bins"]
    bar = Bar()
    bar.add_xaxis(xaxis_data=x_label)
    for l, s1 in self.m.sub_binning_tools.items():
        s = s1[name].  info
        bar.add_yaxis(series_name = l,
                      #category_gap = 0.2,
                      #gap = 0.1,
                      yaxis_data = (s["cnt"] / (s["cnt"].  sum())).tolist(),
                      label_opts = opts.LabelOpts(is_show = False),
                      itemstyle_opts = opts.ItemStyleOpts(opacity = 0.75)
        )

    bar.extend_axis(
        yaxis=opts.AxisOpts(
            name="??????",
            type_="value",
            min_=0,
            #max_=mean_max_lim,
            #interval=5,
            #axislabel_opts=opts.LabelOpts(formatter="{value} "),
        )
    ).set_global_opts(
        legend_opts = opts.LegendOpts(
            pos_top = "5%",
        ),

        title_opts = opts.TitleOpts(
            padding = 3,
            pos_left = "center",
            # title = name,
            # title_textstyle_opts = opts.TextStyleOpts(
            #     font_size = 15,
            #     padding = 3,

            # ),
            # subtitle="???12????????????????????????",
            # subtitle_textstyle_opts = opts.TextStyleOpts(
            #     font_size = 15,
            #     padding = 3,
            #     color = "black",
            # ),

        ),
        tooltip_opts=opts.TooltipOpts(
            is_show=True,
            trigger="axis",
            axis_pointer_type="cross"
        ),
        xaxis_opts = opts.AxisOpts(
            type_="category",
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
            name="??????",
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
    for l, s in _t_info.items():
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
    bar.height = "400px"
    bar.width = "500px"
    grid = Grid()
    #grid.add(bar, grid_opts = opts.GridOpts(pos_top = '50%'))
    #grid.add(line, grid_opts = opts.GridOpts(pos_top = '50%'))
    bar.overlap(line)
    bar.render(template_name = "simple_chart_body.html", path = path)


lgt_dict = sb.sub_binning_tools

for name in sample1.columns:
    _t_info = OrderedDict()
    for label in lgt_dict.keys():
        _info = lgt_dict[label][name].info
        _info["mean"] = _info["mean"].  apply(lambda x:min(x, UPPER))
        _t_info[label] = _info
    binning_html(_t_info, path = "{0}.html". format(name), name = name)

http://127.0.0.1:5005/b_html?name=Lgt2021_0713_1407_1626156435&db=/home/lsy/Project/web/.model_profile/model.db&dir=/home/lsy/Project/web/.model_profile/Lgt2021_0713_1407_1626156435/

import RAW.logitSDK
reload(RAW.logitSDK)
sb.binning_tools["month"].  info
sb.binning_tools["month"].  info
pd.read_excel(sb.binning_excel_path, sheet_name = "1").h1
sb.binning_excel_path
import xlrd
reload(xlrd)

#pd.concat(sb.index_bifurcate, axis = 1)
#/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
import requests
import json
r = requests.post("http://127.0.0.1:5005/cluster", json=_d)
print(r.text)


lc = sb.recorder.load_cluster()
res = lc.cluster(cols=["month"])
res[0]

_d = dict()
_d["dir"] = sb.recorder.dir
_d["db"] = sb.recorder.db
_d["name"] = sb.recorder.name
data = _d
_d["cols"] = ["month"]


import uuid

UUID('bd65600d-8669-4903-8a14-af88203add38')
>>> str(uuid.uuid4())
'f50ec0b7-f960-400d-91f0-c42a6d44e3d0'
>>> uuid.uuid4().hex

mr1 = model_result().load_pattern(pd.read_pickle("/home/lishiyu/Project/model_vision/.model_profile/Lgt2021_0723_1001_1627005664/.cache/7051b0c788964fd6b2b6f03c679380c1.pkl"))
mr1.coef
mr1.result["cols"]

