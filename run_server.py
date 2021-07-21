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
from RAW.logitSDK import loader

HOST = "0.0.0.0"
PORT = 5005


## 建立symlink
app = flask.Flask(__name__, template_folder='./',static_folder="",static_url_path="")
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
    ld = loader(data["name"], data["db"], data["dir"])
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

@app.route("/test", methods=["GET"])
def test():
    return render_template("test.html")

@app.route("/test1", methods=["GET"])
def test1():
    return render_template("test1.html", name = "/home/lsy/Project/web/test.html")

@app.route("/test2", methods=["GET"])
def test2():
    return {"a": 1}

@app.route("/test3", methods=["GET"])
def test3():
    return {"a": 1}

@app.route("/b_html", methods=["POST"])
def get_b_html():
    _d = request.get_json()
    ## print("***************")
    ## print(_d)
    ## print("***************")
    ld = get_loader(_d)
    html_path = pd.Series(ld.load_html_map())
    html_path = html_path.apply(cpath).to_dict()
    cmt = ld.load_table("comment").set_index("index")["comment"].  to_dict()
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
    _df = loader.load_recorder()
    model_info = list()
    for j1, j2 in _df.iterrows():
        _i = j2.copy()
        _i["data"] = [i1 + ": " + str(i2) for i1, i2 in j2.items()]
        model_info.append(_i)
    return render_template('model_repository.html', info = model_info)

@app.route("/mv", methods=["GET"])
def pic():
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
