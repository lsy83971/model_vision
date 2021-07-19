import sys
import os
import math
from RAW.logit import lgt, KS, roc_auc_score, cond_part, var_find
from datetime import datetime, date
import pickle
import warnings
warnings.filterwarnings("ignore")
from default_var import *
from V2.RAW.db import id_concat
from V2.VRB.save_pboc2_concat import concat
import pandas as pd
pd.set_option("display.max_row", 500)


pd.Series(os.listdir(VRB_DATA_DIR)).cc("ZBCASH")
pd.Series(os.listdir(VRB_DATA_DIR)).cc("JD")

#########################################################

def get_xy(
        char = "tianyan_ZBCASH_2020(07|08)-",  # DATA
        y_char = "y_ZBCASH.pkl", # modify
        fields = None,
        rate = 1,
        y_label = "odhis30_first",
        mode = "tianyan",
        l1 = -1,
        l2 = 30,

):
    if mode == "tianyan":
        cmt = read_cmt("cmt_tianyan.pkl")
        if fields is not None:
            fields = ["t." + i for i in fields]
    if isinstance(fields, list):
        if mode == "tianyan":
            if "t.datadt" not in fields:
                fields.append("t.datadt")
            if "t.idcard" not in fields:
                fields.append("t.idcard")
    x = concat(char, rate = rate, fields = fields)



    x.columns = pd.Series(x.columns).apply(lambda x:x.split(".")[ - 1])
    x.r1({"date_time": "dt1", "id_number": "id_card"})
    x.r1({"datadt": "dt1", "idcard": "id_card"})
    x.r1({"t.datadt": "dt1", "t.idcard": "id_card"})

    y = read_raw(y_char)
    y.r1({"c.grant_date": "dt", "b.id_card": "id_card"})
    xy = id_concat(x, y, l1 = l1, l2 = l2, sym = "d.loan_no")
    X = xy[x.columns.tolist() + ["dt"]]
    X = X.replace("", -999999).replace("false", 0).replace("true", 1).fillna(-999998)
    Y = xy[y_label]
    Z = xy[["dt1", "dt", "id_card", "d.loan_no"]]
    return X, Y, Z, cmt


x_ZBCASH, y_ZBCASH, z_ZBCASH, cmt = get_xy(
    char = "tianyan_ZBCASH_2020-(06|07|08|09|10|11)",
    y_char = "y_ZBCASH.pkl",
    y_label = "odhis30_third",
    rate = 0.3)

x_JD, y_JD, z_JD, cmt = get_xy(
    char = "tianyan_JD_2020-(08|09|10|11|12)",
    y_char = "y_JD.pkl",
    y_label = "odhis30_third",
    l2 = 90,
    rate = 1)

#####################################################################

# ZBCASH log_ZBCASH.tsp Lgt20210527_1042
# JD log_JD.tsp Lgt20210527_1222

b_ZBCASH = var_find("Lgt20210527_1042.xlsx", ["1", "2"])
_ind0_ZBCASH = b_ZBCASH[0][(b_ZBCASH[0] < 4e-5) & (b_ZBCASH[0] > 1e-10)]. index
_ind1_ZBCASH = b_ZBCASH[1][(b_ZBCASH[1] < 4e-3) & (b_ZBCASH[1] > 5e-7)]. index
_ind2_ZBCASH = b_ZBCASH[2][(b_ZBCASH[2] > 3e-4)]. index
_ind_ZBCASH = _ind0_ZBCASH.intersection(_ind1_ZBCASH).intersection(_ind2_ZBCASH)

log_ZBCASH = lgt(X = x_ZBCASH[_ind_ZBCASH], Y = y_ZBCASH, cmt = cmt)
log_ZBCASH.binning(quant = 15, ruleB = 8,
                   ruleV = 5000, stable_points = [- 1],
                   mode = "b", cols = None, remain_tick = False)
log_ZBCASH.update_woe()
cols_ZBCASH = log_ZBCASH.var(rule = 0.99).index
_ind_JD.union(cols_ZBCASH)

b_JD = var_find("Lgt20210527_1222.xlsx", ["1", "2"])
_ind0_JD = b_JD[0][(b_JD[0] < 3e-4) & (b_JD[0] > 1e-10)]. index
_ind1_JD = b_JD[1][(b_JD[1] < 3e-3) & (b_JD[1] > 5e-7)]. index
_ind2_JD = b_JD[2][(b_JD[2] > 5e-5)]. index
_ind_JD = _ind0_JD.intersection(_ind1_JD).intersection(_ind2_JD)



log_JD = lgt(X = x_JD[_ind_JD], Y = y_JD, cmt = cmt)
log_JD.binning(quant = 15, ruleB = 5,
               ruleV = 5000, stable_points = [- 1],
               mode = "b", cols = None, remain_tick = True)
log_JD.update_woe()
cols_JD = log_JD.var(rule = 0.99).index
cols_ALL = cols_JD.union(cols_ZBCASH)
pd.Series(cols_ALL).to_csv("/home/bozb/lsy/DATA/MODEL/SELECTED_COLS.csv")

####################################################

## Lgt20210528_1620_modelresult.pkl
ss = lgt.load_model_result("Lgt20210528_1620_modelresult.pkl")
ss1 = ss["trans_func"](x_JD[ss["cols"]]. replace("", -999999).fillna(-1). astype(float))
score1 = pd.Series(ss["model"]. predict_proba(ss1[ss["cols"]])[:, 1], name = "score", index = y_JD.index)
score2 = pd.concat([score1, y_JD], axis = 1)
score2.binning("score", "odhis30_third", func = lambda y:{"cnt": y.shape[0], "mean": y.mean()}, quant = 10, mode = "v")

ss1 = ss["trans_func"](x_ZBCASH[ss["cols"]]. replace("", -999999).fillna(-1). astype(float))
score1 = pd.Series(ss["model"]. predict_proba(ss1[ss["cols"]])[:, 1], name = "score", index = y_ZBCASH.index)
score2 = pd.concat([score1, y_ZBCASH], axis = 1)
score2.binning("score", "odhis30_third", func = lambda y:{"cnt": y.shape[0], "mean": y.mean()}, quant = 10, mode = "v")
KS(score2["odhis30_third"], score2["score"])

####################################################


x_JD["channel"] = "JD"
x_ZBCASH["channel"] = "ZBCASH"

x = pd.concat([x_JD[cols_ALL.tolist() + ["channel"]], x_ZBCASH[cols_ALL.tolist() + ["channel"]]])
y = pd.concat([y_JD, y_ZBCASH])
z = pd.concat([z_JD, z_ZBCASH])
w = pd.concat([x, y, z], axis = 1)
w.to_csv("XCJD_TY_V1_sample.csv", index = False)
cond = (z["dt"] < datetime(2021, 1, 20))

cond_JD = x["channel"] == "JD"
cond_od = y == 1
cond_ZBCASH = x["channel"] == "ZBCASH"

x1 = x[cond]
y1 = y[cond]
z1 = z[cond]

x2 = pd.concat([x1] + [x[cond & cond_JD & cond_od]] * 3, axis = 0)
y2 = pd.concat([y1] + [y[cond & cond_JD & cond_od]] * 3, axis = 0)
z2 = pd.concat([z1] + [z[cond & cond_JD & cond_od]] * 3, axis = 0)



log2 = lgt(X = x2, Y = y2, cmt = cmt)
log2.binning(quant = 15, ruleB = 15,
                   ruleV = 10000, stable_points = [- 1],
                   mode = "b", cols = None, remain_tick = False)
sample = cond_part((x2["channel"] == "JD"),[0.3])
log2.draw_binning(conds = sample, upper_lim = 0.08)
log2.draw_binning_excel()
b = var_find(log2.binning_excel_path, ["1", "2"])
_ind0 = b[0][(b[0] < 5e-4) & (b[0] > 1e-10)]. index
_ind1 = b[1][(b[1] < 2e-1) & (b[1] > 5e-7)]. index
_ind2 = b[2][(b[2] > 5e-4)]. index
_ind = _ind0.intersection(_ind1).intersection(_ind2)
log2.update_woe(cols = _ind.tolist() + ["channel"])
cols = log2.var(cols = _ind,
                  rule = 0.4,
                  exclud = ["loan_credit_limit_amt",
                            "left_cnt_cash_m36_mean",
                  ],
).index

cond_ZBCASH = x2["channel"] == "ZBCASH"
cond_ZBCASH_dt = z2["dt"] < z2[cond_ZBCASH]["dt"]. quantile(0.7)
cond_JD = x2["channel"] == "JD"
cond_JD_dt = z2["dt"] < z2[cond_JD]["dt"]. quantile(0.7)
cond_old = (cond_ZBCASH & cond_ZBCASH_dt) | (cond_JD & cond_JD_dt)
cond_new = (~cond_old)
train_sample = [cond_old.values, cond_new.values]


log2.train(cols = cols, sample = train_sample,
                 labels = ["train", "test"], quant = 10,
                 C = 0.1,
                 rule = 0.3,
)
pd.Series(log2.model.coef_[0], index = log2.model_result["cols"])
log2.save_model_report()
log2.save_model_result()


##############################################################


y_JD_raw = read_raw("y_JD.pkl")[["d.loan_no", "b.id_card", "c.grant_date"]]
y_JD_raw. r1({"d.loan_no": "loan_no",
              "b.id_card": "id_card",
              "c.grant_date": "dt"
})
y_JD_raw_first = y_JD_raw.sort_values("dt").drop_duplicates("id_card")[["id_card", "dt"]]
y_JD_raw_first.r1({"dt":"first_dt"})
y_JD_merge = y_JD_raw.merge(y_JD_raw_first, how = "left", on = "id_card")
y_JD_merge["span"] = (y_JD_merge["dt"] - y_JD_merge["first_dt"]).dt.days


y_ZBCASH_raw = read_raw("y_ZBCASH.pkl")[["d.loan_no", "b.id_card", "c.grant_date"]]
y_ZBCASH_raw. r1({"d.loan_no": "loan_no",
              "b.id_card": "id_card",
              "c.grant_date": "dt"
})
y_ZBCASH_raw_first = y_ZBCASH_raw.sort_values("dt").drop_duplicates("id_card")[["id_card", "dt"]]
y_ZBCASH_raw_first.r1({"dt":"first_dt"})
y_ZBCASH_merge = y_ZBCASH_raw.merge(y_ZBCASH_raw_first, how = "left", on = "id_card")
y_ZBCASH_merge["span"] = (y_ZBCASH_merge["dt"] - y_ZBCASH_merge["first_dt"]).dt.days

old_cust_loan_no = pd.concat([y_JD_merge["loan_no"][(y_JD_merge["span"] > 30)],
                              y_ZBCASH_merge["loan_no"][(y_ZBCASH_merge["span"] > 30)]])

new_cust_loan_no = pd.concat([y_JD_merge["loan_no"][(y_JD_merge["span"] <= 30)],
                              y_ZBCASH_merge["loan_no"][(y_ZBCASH_merge["span"] <= 30)]])


#################################

b = var_find("Lgt20210608_1745.xlsx", ["1", "2"])
_ind0 = b[0][(b[0] < 5e-4) & (b[0] > 1e-10)]. index
_ind1 = b[1][(b[1] < 2e-1) & (b[1] > 5e-7)]. index
_ind2 = b[2][(b[2] > 5e-4)]. index

_ind = _ind0.intersection(_ind1).intersection(_ind2)

log2 = lgt(X = x2[_ind], Y = y2, cmt = cmt)
log2.binning(quant = 15, ruleB = 15,
                   ruleV = 10000, stable_points = [- 1],
                   mode = "b", cols = None, remain_tick = False)
sample = cond_part((x2["channel"] == "JD"),[0.3])
log2.draw_binning(conds = sample, upper_lim = 0.08)
log2.draw_binning_excel()
_ind = _ind0.intersection(_ind1).intersection(_ind2)
log2.update_woe(cols = _ind.tolist())
cols = log2.var(cols = _ind,
                  rule = 0.4,
                  exclud = ["loan_credit_limit_amt",
                            "left_cnt_cash_m36_mean",
                  ],
).index

cond_ZBCASH = x2["channel"] == "ZBCASH"
cond_ZBCASH_dt = z2["dt"] < z2[cond_ZBCASH]["dt"]. quantile(0.7)
cond_JD = x2["channel"] == "JD"
cond_JD_dt = z2["dt"] < z2[cond_JD]["dt"]. quantile(0.7)
cond_old = (cond_ZBCASH & cond_ZBCASH_dt) | (cond_JD & cond_JD_dt)
cond_new = (~cond_old)
train_sample = [cond_old.values, cond_new.values]

log2.train(cols = _ind, sample = train_sample,
                 labels = ["train", "test"], quant = 10,
                 C = 0.1,
                 rule = 0.3,
)
log2.train(cols = cols, sample = train_sample,
                 labels = ["train", "test"], quant = 10,
                 C = 0.1,
                 rule = 0.3,
)
pd.Series(log2.model.coef_[0], index = log2.model_result["cols"])
log2.save_model_report()
log2.save_model_result()


##################################
x_old = x2[z2["d.loan_no"]. isin(old_cust_loan_no)]
y_old = y2[z2["d.loan_no"]. isin(old_cust_loan_no)]
z_old = z2[z2["d.loan_no"]. isin(old_cust_loan_no)]

log2 = lgt(X = x_old, Y = y_old, cmt = cmt)
log2.binning(quant = 15, ruleB = 6,
                   ruleV = 3000, stable_points = [- 1],
                   mode = "b", cols = None, remain_tick = False)
sample = cond_part((x_old["channel"] == "JD"),[0.3])
log2.draw_binning(conds = sample, upper_lim = 0.08)
log2.draw_binning_excel()

b = var_find(log2.binning_excel_path, ["1", "2"])
_ind0 = b[0][(b[0] < 5e-4) & (b[0] > 1e-10)]. index
_ind1 = b[1][(b[1] < 2e-1) & (b[1] > 5e-7)]. index
_ind2 = b[2][(b[2] > 5e-4)]. index
_ind = _ind0.intersection(_ind1).intersection(_ind2)

log2.update_woe(cols = _ind.tolist() + ["channel"])
cols = log2.var(cols = _ind,
                  rule = 0.4,
                  exclud = ["loan_credit_limit_amt",
                            "left_cnt_cash_m36_mean",
                  ],
).index
cols = cols.drop(["loan_past_quota5w_cnt", "credit_total_amt", "loan_unpaid_consum_pct"])

cond_ZBCASH = x_old["channel"] == "ZBCASH"
cond_ZBCASH_dt = z_old["dt"] < z_old[cond_ZBCASH]["dt"]. quantile(0.7)

cond_JD = x_old["channel"] == "JD"
cond_JD_dt = z_old["dt"] < z_old[cond_JD]["dt"]. quantile(0.7)

cond_old = (cond_ZBCASH & cond_ZBCASH_dt) | (cond_JD & cond_JD_dt)
cond_new = (~cond_old)
train_sample = [cond_old.values, cond_new.values]


log2.train(cols = cols, sample = train_sample,
                 labels = ["train", "test"], quant = 10,
                 C = 0.1,
                 rule = 0.3,
)

pd.Series(log2.model.coef_[0], index = log2.model_result["cols"])
log2.save_model_report()
log2.save_model_result()



##################################



x_new = x2[z2["d.loan_no"]. isin(new_cust_loan_no)]
y_new = y2[z2["d.loan_no"]. isin(new_cust_loan_no)]
z_new = z2[z2["d.loan_no"]. isin(new_cust_loan_no)]







self = be


index = "queryrecord_loanapproval_l1m_l24m_pct"
(infos["bad"] + 0.5)
import RAW.logit
reload(RAW.logit)
from RAW.logit import binning_excel


def mean_dif_ent(self, index):
    from RAW.db import db_ent
    infos = []
    std_rate = self.total[index]["bad_cnt"]. sum() / self.total[index]["总数"]. sum()
    for l, k in enumerate(self.labels):
        m = self.info[k][index]["总数"]
        info = self.info[k][index][["坏率"]]
        info["label"] = k
        infos.append(info)
        pd.Series(index = m.index)
        if l == 0:
            m0 = pd.Series(1, index = m.index)
        m0 *= m
    m0 = pd.Series(m0 ** (1 / len(self.labels)), name = "cnt")
    for i in infos:
        i["cnt"] = m0

    infos = pd.concat(infos).reset_index()
    infos["bad"] = infos["坏率"] * infos["cnt"] + 100 * (std_rate)
    infos["good"] = infos["cnt"] - infos["bad"] + 100 * (1 - std_rate)
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










from RAW.logit import binning_excel
be = binning_excel(log = log2)
be.total["queryrecord_loanapproval_l1m_l24m_pct"]
be.info["1"]["queryrecord_loanapproval_l1m_l24m_pct"]
be.info["2"]["queryrecord_loanapproval_l1m_l24m_pct"]


print("lending_agency_uncleared_num" in _ind2)
print("S_ZB_VB_MARRIAGE" in _ind2)
print("N_PB_VD_JYDNHDSUM" in _ind2)


"N_BRJDYX_VB_ALSM6CELLCAONORGNUM" in _ind
"native_credit_query_log_dkspcs_6" in _ind1
b[1]. loc["native_credit_query_log_dkspcs_6"]
b[2]. loc["S_ZB_VB_MARRIAGE"]

b[0]["queryrecord_loanapproval_l1m_l24m_pct"]
b[0]["N_BRJDYX_VB_ALSM1IDCAOFFORGNUM"]


import pandas as pd
sample = pd.read_csv("XCJD_TY_V1_sample.csv")
sample1 = sample[[
    "card_cm_bank12m_pct",
    "card_cm_bank24m_pct",
    "card_cm_bank60m_pct",
]]
y = sample[["odhis30_third"]]
sample1 = pd.read_csv("./py/sample1.csv")
y = pd.read_csv("./py/y.csv")["odhis30_third"]

y = sample[["odhis30_third"]]

from importlib import reload
import RAW.int
import RAW.logit

reload(RAW.logit)
reload(RAW.int)
from RAW.logit import lgt, intLDict, cond_part
sb = lgt(X = sample1, Y = y)
sb.binning(mode = "C", ruleC = -0.0001)
s1, s2 = cond_part(pd.Series(sample1.index.astype(str)).apply(lambda x:x[ - 1]).astype(float), 0.5)
conds = [s1, s2]
subs = sb.draw_binning([s1, s2])
self = sb


sb.binning_tools["card_cm_bank12m_pct"].  info.T
sb.label = "A"
sb.part = "A"
i = "card_cm_bank12m_pct"


_df = lgt.binning_barplot_df([sb], i)
_df_raw = sb.binning_tools[i].  info
_df_raw["bins"].  tolist()

x_label =

bar = (
    Bar()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="蒸发量",
        yaxis_data=[
            2.0,
            4.9,
            7.0,
            23.2,
            25.6,
            76.7,
            135.6,
            162.2,
            32.6,
            20.0,
            6.4,
            3.3,
        ],
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        series_name="降水量",
        yaxis_data=[
            2.6,
            5.9,
            9.0,
            26.4,
            28.7,
            70.7,
            175.6,
            182.2,
            48.7,
            18.8,
            6.0,
            2.3,
        ],
        label_opts=opts.LabelOpts(is_show=False),
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="温度",
            type_="value",
            min_=0,
            max_=25,
            interval=5,
            axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
        )
    )
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            name_rotate = 45,
            type_="category",
            axislabel_opts = opts.LabelOpts(rotate=45),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="水量",
            type_="value",
            min_=0,
            max_=250,
            interval=50,
            axislabel_opts=opts.LabelOpts(formatter="{value} ml"),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    )
)

line = (
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="平均温度",
        yaxis_index=1,
        y_axis=[2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2],
        label_opts=opts.LabelOpts(is_show=False),
    )
)




sample1.to_csv("sample1.csv", index = False)
y.to_csv("y.csv", index = False)
