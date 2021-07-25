# -*- coding: utf-8 -*-
"""
@author: lishiyu
"""

import sys
from flask import render_template
sys.path.append("./py")
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import pickle
import pandas as pd
import math
import logging
import time
import json
import datetime
from dateutil import tz
import pandas as pd
import numpy as np
import re
from datetime import datetime
import flask
from flask import request
import traceback
import os
from RAW.logitSDK import recorder, cond_part, step_train, model_result
from RAW.recorder import echarts_plot

from RAW.int import binning
import uuid
HOST = "0.0.0.0"
PORT = 5005


## 建立symlink
app = flask.Flask(__name__, template_folder='./',static_folder="",static_url_path="")
def jinja_uuid():
    return uuid.uuid4().hex
app.jinja_env.globals.update(jinja_uuid=jinja_uuid)

_cwd = os.path.dirname(os.path.abspath(__file__))
_dst = _cwd + "/root"
_src = "/"
if not os.path.exists(_dst):
    os.symlink(_src, _dst)
def cpath(abspath):
    return "root" + abspath

app.config['JSON_AS_ASCII'] = False
models = {}
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')  # 设置日志输出格式
loghanlder = logging.FileHandler("./logger", encoding='utf-8')
loghanlder.setFormatter(formatter)
loghanlder.setLevel(logging.INFO)
logger.addHandler(loghanlder)
logger.setLevel(logging.INFO)

def get_loader(data):
    ld = recorder().from_name(data["name"], data["db"], data["dir"])
    return ld

@app.route("/save_selected", methods=["POST"])
def save_selected():
    _d = request.get_json()
    ld = get_loader(_d)
    ld.save_cols(_d["selected_index"], symbol = _d["symbol"])
    print("***************")
    print(_d)
    print("***************")
    return {"a": 1}

@app.route("/save_cluster", methods=["POST"])
def save_cluster():
    _d = request.get_json()
    ld = get_loader(_d)
    ld.save_cluster(_d["cluster_index"], symbol = _d["symbol"])
    print(_d)
    return {"a": 1}

@app.route("/cluster", methods=["POST"])
def cluster():
    _d = request.get_json()
    print(_d)
    ld = get_loader(_d)
    cl = ld.load_cluster()
    cmt = ld.load_comment().  to_dict()
    html_path = pd.Series(ld.load_html_map())
    html_path = html_path.apply(cpath).to_dict()
    try:
        bif_mean = ld.load_table("bifurcate").set_index("index")["bif_mean"].  to_dict()
        bif_porp = ld.load_table("bifurcate").set_index("index")["bif_porp"].  to_dict()
        bif_ent = ld.load_table("bifurcate").set_index("index")["bif_ent"].  to_dict()
    except:
        bif_mean = dict()
        bif_porp = dict()
        bif_ent = dict()

    _d1 = dict()
    for i in ["cols", "ruleD", "ruleN"]:
        if i in _d:
            _d1[i] = _d[i]
    res = cl.cluster(**_d1)
    cluster_res = [(j, str(int(j)), i.to_dict()) for j, i in enumerate(res)]
    #def render_template_string(source, **context):
    return render_template('cluster.html',
                           html_path = html_path,
                           cmt = cmt,
                           bif_mean = bif_mean,
                           bif_porp = bif_porp,
                           bif_ent = bif_ent,
                           cluster_res=cluster_res,
    )

@app.route("/train", methods=["POST"])
def train():
    #_d={"train_cols":sb.woevalue.columns.tolist(),"train_split_quant":0.7,"C":0.1}    
    _d = request.get_json()
    ld = get_loader(_d)
    xy = ld.load_woe_x(_d["train_cols"] + ["label", "dt"])
    x = xy.drop(["label", "dt"], axis=1)
    y = xy["label"]
    dt = pd.to_datetime(xy["dt"])
    ent = ld.load_ent()
    cmt = ld.load_comment()
    sample_cond = cond_part(dt, float(_d["train_split_quant"]))
    sample_cond_dict = {"train": sample_cond[0], "test": sample_cond[1]}
    res = step_train(x.loc[sample_cond[0]],
                     y.loc[sample_cond[0]],
                     ent, mode="l1",
                     C=float(_d["train_c"]), 
    )
    model = res["model"]
    cols = res["cols"]

    x1 = x[cols]
    quant = 10
    score = pd.Series(model.predict_proba(x1)[:, 1], index = x1.index)
    standard_woe = math.log(((y == 1).sum() + 0.5) / ((y == 0).sum() + 0.5))    
    ticks = binning.tick(
        x = score, quant = quant,
        single_tick = False,
        ruleV = score.shape[0] / quant / 2,
        mode = "v"). ticks_boundaries()
    mr = model_result().from_result(
        model = model,
        cols = cols,
        ticks = ticks,
        standard_woe = standard_woe,
        binning_tools = ld.load_binning_tools(),
        train_cond = sample_cond_dict, 
    )
    t_info = dict()
    t_info["train"] = mr.binning_sample(x.loc[sample_cond[0], cols], y.loc[sample_cond[0]])
    t_info["test"] = mr.binning_sample(x.loc[sample_cond[1], cols], y.loc[sample_cond[1]])
    api_info = {i: j["binning"] for i, j in t_info.items()}
    api_info["train"]. r1({"总数": "cnt", "提升": "mean"})
    api_info["test"]. r1({"总数": "cnt", "提升": "mean"})
    x_label = api_info["test"]. index.tolist()
    bar = echarts_plot(x_label, api_info, UPPER=5)
    bar_html = bar.render_embed(template_name = "simple_chart_body.html")
    index_info = pd.concat(
        [ent.loc[cols], 
         mr.coef,
         cmt], join="inner", axis=1).reset_name("指标")
    index_info.r1({"ent": "区分度", "comment": "中文名"})
    index_info["区分度"] = index_info["区分度"]. apply(lambda x:round(x, 6))
    index_info["系数"] = index_info["系数"]. apply(lambda x:round(x, 6))    
    result_info = pd.DataFrame(t_info).drop("binning")
    result_info = result_info.reset_name("")
    result_info["train"] = result_info["train"]. apply(lambda x:round(x, 4))
    result_info["test"] = result_info["test"]. apply(lambda x:round(x, 4))
    result_path = uuid.uuid4().hex + ".pkl"
    if not os.path.exists(ld.cache):
        os.mkdir(ld.cache)
    mr.save(ld.cache + result_path)
    return render_template('train_result.html',
                           bar_html = bar_html,
                           index_info = index_info, 
                           result_info = result_info,
                           result_path = result_path, 
    )

@app.route("/save_result", methods=["POST"])
def save_result():
    _d = request.get_json()
    ld = get_loader(_d)
    mr_path = ld.cache + _d["mr_path"]
    mr = model_result().load_pattern(pd.read_pickle(mr_path))
    _id = str(datetime.now().strftime("%Y%m%d_%H%M%S_%s"))
    report_path = ld.saved_result + "modelreport_{0}.xlsx". format(_id)
    result_path = ld.saved_result + "modelresult_{0}.pkl". format(_id)        
    ld.r_model_report(mr, path=report_path)
    os.system("cp {0} {1}". format(mr_path, result_path))
    return {"a": 1}

@app.route("/b_html", methods=["POST"])
def get_b_html():
    _d = request.get_json()
    ld = get_loader(_d)
    html_path = pd.Series(ld.load_html_map())
    html_path = html_path.apply(cpath).to_dict()
    cmt = ld.load_comment().  to_dict()
    try:
        bif_mean = ld.load_table("bifurcate").set_index("index")["bif_mean"].  to_dict()
        bif_porp = ld.load_table("bifurcate").set_index("index")["bif_porp"].  to_dict()
        bif_ent = ld.load_table("bifurcate").set_index("index")["bif_ent"].  to_dict()
        names = sorted(html_path, key=lambda x:bif_ent.get(x, 0))
    except:
        raise

    return render_template('b_charts.html',
                           names = names,
                           html_path = html_path,
                           cmt = cmt,
                           bif_mean = bif_mean,
                           bif_porp = bif_porp,
                           bif_ent = bif_ent,
    )

@app.route("/model_repository", methods=["GET"])
def model_repository():
    _df = recorder.load_recorder()
    model_info = list()
    for j1, j2 in _df.iterrows():
        _i = j2.copy()
        _i["data"] = [i1 + ": " + str(i2) for i1, i2 in j2.items()]
        model_info.append(_i)
    return render_template('model_repository.html', info = model_info)

@app.route("/mv", methods=["GET"])
def mv():
    return render_template('index.html')


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(
        host=HOST,
        port=PORT,
        debug=True
    )
    ############ 修改template

