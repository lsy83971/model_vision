import json
from copy import deepcopy
import numbers
import inspect
from scipy import stats
import math
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
from datetime import datetime
import math

def reset_name(self, name, inplace = False):
    """
    将index命名为name 作为新的一列赋值添加到dataFrame第一列
    """
    self.index.name = name
    return self.reset_index(inplace = inplace)

pd.DataFrame.reset_name = reset_name
pd.Series.reset_name = reset_name


def arg_wrapper(f):
    """
    arg_wrapper会将函数执行时多余的入参进行剔除
    使得多余的入参不影响函数的执行
    例如f=lambda a,b:a+b
    执行f(a=1,b=2,c=3)会报错
    arg_wrapper(f)(a=1,b=2,c=3) 则能够正常运行
    """
    def f1(**kwargs):
        kwargs1 = dict()
        for i in list(inspect.signature(f).parameters.keys()):
            if i in kwargs:
                kwargs1[i] = kwargs[i]
            else:
                kwargs1[i] = None
        return f(**kwargs1)
    return f1

def center_dev(mean_old, std_old, cnt_old, mean_new, std_new, cnt_new):
    """
    obj_func of old sample and new sample
    with old sample contains new sample
    """
    obj = 0
    obj = (mean_new - mean_old) * ((cnt_new/ (cnt_old - cnt_new + 10))**(1 / 2)) / std_old * (cnt_old ** (1 / 2))
    if mean_old < mean_new:
        obj = -obj
    _score = stats.norm.cdf(obj)
    return _score

def double_cross_entrophy_dev(x):
    """
    tools to calculate cross_entrophy
    """
    if x == 0:
        return 0
    return x * math.log(x)

class BoundError(Exception):
    pass

class cutbin(type):
    """
    init: update
    fit: fit + update + check_valid
    """
    def __new__(cls,  name,  bases,  attrs):
        """
        _dec_tables_update:        函数执行完毕后进行update
        update_decorator:          对_dec_tables_update中函数进行装饰
        _dec_tables_check：        函数执行完毕后进行checkvalid
        checkvalid_decorator:      对_dec_tables_check中函数进行装饰
        class在执行 __new__以后会检查
        attr_init, attr_init_default, fit, merge, check_valid
        这些函数或属性是否有定义 若无则补定义为默认值
        attr->bin_attr: 箱体属性
        attr->update: 将字典添加到bin_attr 和 __dict__
        attr->fit: 使用checkvalid装饰 & update装饰
        attr->merge: 使用update装饰
        attr->__init__: 使用checkvalid装饰
        attr->__init__:
             标准化初始化函数 无需在类中定义__init__
             作为替代 定义cls.attr_init(必填字段)
                          cls.attr_init_default(默认值字段)即可
             初始化函数会将入参全部赋值为对象的属性
             检查cls.attr_init是否齐全 否则报错
             缺失的cls.attr_init_default使用默认值补全
        attr->info: 返回bin_attr中的简单类型字典

        """

        # _dec_tables_update = ["fit", "merge"]
        # _dec_tables_check = ["fit", "__init__"]
        # _rename_tables = list(set(_dec_tables_update + _dec_tables_check))

        if "update" not in attrs:
            attrs["update"] = lambda self, kwargs:[self.bin_attr.update(kwargs),
                                              self.__dict__. update(kwargs)]

        def attr_init(self, *args, **kwargs):
            self.bin_attr = dict()
            _l1 = self.attr_init + list(self.attr_init_default.keys())
            try:
                for k, i in enumerate(args):
                    kwargs[_l1[k]] = i
            except:
                raise Exception("args more than ")
            for i in self.attr_init:
                assert i in kwargs
            _d = self.attr_init_default.copy()
            _d.update(kwargs)
            self.update(_d)

        attrs["__init__"] = attr_init

        def check_simple_type(_d):
            _d1 = dict()
            for i, j in _d.items():
                if isinstance(j, numbers.Number) or isinstance(j, int) or isinstance(j, str):
                    _d1[i] = j
            return _d1

        @property
        def info(self):
            _d = check_simple_type(self.bin_attr)
            _d["bins"] = self.word
            return _d

        attrs["info"] = info
        _type = type.__new__(cls,  name,  bases,  attrs)

        for i in ["attr_init"]:
            if i not in dir(_type):
                setattr(_type, i, [])

        for i in ["attr_init_default"]:
            if i not in dir(_type):
                setattr(_type, i, dict())

        for i in ["fit", "merge"]:
            if i not in dir(_type):
                setattr(_type, i, lambda self, * x, **y:dict())

        for i in ["check_valid"]:
            if i not in dir(_type):
                setattr(_type, i, lambda self, * x, **y:None)

        def update_decorator(f):
            if not hasattr(f, "dec_set"):
                f.dec_set = set()
            if "update" in f.dec_set:
                return f
            def f1(self, *args, **kwargs):
                _res = f(self, *args, **kwargs)
                self.update(_res)
                return _res
            f1.dec_set = f.dec_set
            f1.dec_set.add("update")
            return f1

        def checkvalid_decorator(f):
            if not hasattr(f, "dec_set"):
                f.dec_set = set()
            if "valid" in f.dec_set:
                return f
            def f1(self, *args, **kwargs):
                _res = f(self, *args, **kwargs)
                self.check_valid()
                return _res
            f1.dec_set = f.dec_set
            f1.dec_set.add("valid")
            return f1

        _dec_tables_update = ["fit", "merge"]
        _dec_tables_check = ["fit", "__init__"]

        for i in _dec_tables_update:
            try:
                setattr(_type, i, update_decorator(getattr(_type, i)))
            except:
                pass

        # for i in _dec_tables_check:
        #     try:
        #         setattr(_type, i, checkvalid_decorator(getattr(_type, i)))
        #     except:
        #         pass

        return _type

class interval(object,  metaclass = cutbin):
    """
    区间类
    包括连续区间型和枚举型
    calculate: 自定义func 计算结果
    fit: 使用类自有func 计算结果
    fit_cond：落在区间段的样本

    merge之前需要预先fit
    """
    def fit_cond(self, x):
        pass

    def fit_df(self, x, df):
        return df.loc[self.fit_cond(x)]

    def fit_kw(self, raw_kw, func):
        """
        x,y,df 为特殊参数
        calculate操作之前
        对于特殊参数
        筛选为落在区间段内的样本执行func
        """
        kw = interval.fit_func_argsfilter(func, raw_kw)
        change_kw_list = ["x", "y", "df"]
        _kw = {i: j for i, j in kw.items() if i not in change_kw_list}
        _kw1 = {i: j for i, j in kw.items() if i in change_kw_list}
        if len(_kw1) > 0:
            _cond = self.fit_cond(self.fit_type(raw_kw["x"]))
            for i, j in _kw1.items():
                _kw1[i] = j.loc[_cond]
        return { ** _kw, **_kw1}

    @staticmethod
    def fit_func_argsfilter(func, kwargs):
        _l = list(inspect.signature(func).parameters.keys())
        return {i: j for i, j in kwargs.items() if i in _l}

    @staticmethod
    def fit_type(x):
        return x

    def is_stick(self):
        if hasattr(self, "_stick"):
            if self._stick == True:
                return True
        return False

    def stick(self):
        self._stick = True

    def calculate(self, func = None, update = True, **kwargs):
        """
        update: does the result update to info
        """
        if func is None:
            func = self.fit_func
        _kw = self.fit_kw(raw_kw = kwargs, func = func)
        _kw["self"] = self
        func = arg_wrapper(func)
        res = func(**_kw)
        if update:
            self.update(res)
        return res

    def fit(self, **kwargs):
        kwargs["func"] = self.fit_func
        return self.calculate(**kwargs)

    @property
    def dev(self, f):
        return 0

    def merge(self, f):
        return dict()

    def reset_info(self):
        """
        delete information derived on x, y ...
        remain information of boundary left right ...
        """
        l1 = self.attr_init
        l2 = list(self.attr_init_default.keys())
        l3 = list(set(l1) | set(l2))
        p = list(self.bin_attr.keys()).copy()
        for i in p:
            if i in l3:
                continue
            del self.bin_attr[i]
            if i in self.__dict__:
                del self.__dict__[i]

class intv(interval):
    """
    连续型区间类
    作为所有数值型区间类的父类
    提供基础方法:
    1. 判断两个区间是否相邻
    2. 区间合并
    3. fit样本：x落在区间的数量
    """
    attr_init = ["left", "right"]
    attr_init_default = OrderedDict({"lt": "close", "rt": "open"})
    def check_valid(self):
        if self.left > self.right:
            raise BoundError("left bound larger than right bound")
        if self.left == self.right:
            if not ((self.lt == "close") and (self.rt == "close")):
                raise BoundError("equal bound not close")

    @staticmethod
    def fit_func(x, **kwargs):
        return {"cnt": x.shape[0]}

    @staticmethod
    def fit_type(x):
        return x.astype(float)

    def fit_cond(self, x):
        """
        数列落在区间 True False
        """
        if self.lt == "close":
            cond1 = (x >= self.left)
        else:
            cond1 = (x > self.left)
        if self.rt == "close":
            cond2 = (x <= self.right)
        else:
            cond2 = (x < self.right)
        return (cond1 & cond2)

    def rightExact(self, f):
        """
        f on the right
        """
        ex = False
        if f.left == self.right:
            if (f.lt == "close") and (self.rt == "open"):
                ex = True
            if (f.lt == "open") and (self.rt == "close"):
                ex = True
        return ex

    def leftExact(self, f):
        """
        f on the left
        """
        ex = False
        if self.left == f.right:
            if (self.lt == "close") and (f.rt == "open"):
                ex = True
            if (self.lt == "open") and (f.rt == "close"):
                ex = True
        return ex

    def basic_merge(self, f):
        _d = dict()
        if self.rightExact(f):
            ex = "right"
        elif self.leftExact(f):
            ex = "left"
        else:
            raise Exception("merge intervals not exact")

        _d["left"] = min(self.left, f.left)
        _d["right"] = max(self.right, f.right)

        if ex == "right":
            _d["rt"] = f.rt
        if ex == "left":
            _d["lt"] = f.lt
        return _d

    def extra_merge(self, f):
        _d = dict()
        try:
            _d["cnt"] = self.cnt + f.cnt
        except:
            pass
        return _d

    def merge(self, f):
        return { ** self.basic_merge(f), ** self.extra_merge(f)}

    @property
    def word(self):
        if self.lt == "close":
            _l = "["
        else:
            _l = "("

        if self.rt == "close":
            _r = "]"
        else:
            _r = ")"
        return _l + "{0:.4f}". format(self.left) + ", " + \
               "{0:.4f}". format(self.right) + _r

    def dev(self, f):
        if (self.is_stick() | f.is_stick()):
            return math.inf
        return self.cnt + f.cnt

    def iszero(self):
        return self.cnt == 0

class intb(intv):
    """
    interval_binary
    y为01类型
    """
    @staticmethod
    def fit_func(y, **kwargs):
        _d = dict()
        _d["good_cnt"] = (y == 0).sum()
        _d["bad_cnt"] = (y == 1).sum()
        _d["cnt"] = _d["good_cnt"] + _d["bad_cnt"]
        _d["mean"] = _d["bad_cnt"] / _d["cnt"]
        _d["woe"] = math.log((_d["bad_cnt"] + 0.5) / (_d["good_cnt"] + 0.5))
        return _d

    def extra_merge(self, f):
        """
        合并区间用
        """
        _d = dict()
        _d["good_cnt"] = self.good_cnt + f.good_cnt
        _d["bad_cnt"] = self.bad_cnt + f.bad_cnt
        _d["cnt"] = self.cnt + f.cnt
        _d["mean"] = _d["bad_cnt"] / _d["cnt"]
        _d["woe"] = math.log((_d["bad_cnt"] + 0.5) / (_d["good_cnt"] + 0.5))

        return _d

    def dev(self, f):
        if (self.is_stick() | f.is_stick()):
            return math.inf

        s = double_cross_entrophy_dev
        _g1 = self.good_cnt
        _b1 = self.bad_cnt
        _c1 = _g1 + _b1
        _g2 = f.good_cnt
        _b2 = f.bad_cnt
        _c2 = _g2 + _b2
        _g3 = _g1 + _g2
        _b3 = _b1 + _b2
        _c3 = _c1 + _c2
        _e1 = s(_g1) + s(_b1) - s(_c1)
        _e2 = s(_g2) + s(_b2) - s(_c2)
        _e3 = s(_g3) + s(_b3) - s(_c3)
        return _e1 + _e2 - _e3

class intc(intv):
    """
    y为浮点类型
    统计cnt var mean
    """
    @staticmethod
    def fit_func(x, y, **kwargs):
        _cnt = x.shape[0]
        if _cnt <= 1:
            _var = 0
        else:
            _var = y.var()

        if _cnt <= 0:
            _mean = 0
        else:
            _mean = y.mean()
        return {
            "cnt": _cnt,
            "mean": _mean,
            "var": _var,
        }

    def extra_merge(self, f):
        """
        合并区间用
        """
        _mean = (self.cnt * self.mean + f.cnt * f.mean) / (self.cnt + f.cnt)
        _var1 = (self.mean - _mean)**2 * self.cnt
        _var2 = (f.mean - _mean)**2 * f.cnt
        _var = (_var1 + _var2 + self.var * self.cnt + f.var * f.cnt) / (self.cnt + f.cnt)
        _cnt = self.cnt + f.cnt
        return {"var": _var, "mean": _mean, "cnt": _cnt}

    def dev(self, f):
        if (self.is_stick() | f.is_stick()):
            return math.inf

        _d = self.extra_merge(f)
        return - center_dev(_d["mean"], _d["var"] ** (1 / 2), _d["cnt"],
                            self.mean, self.var ** (1 / 2), self.cnt)

class binstrv(interval):
    """
    区间类
    """
    attr_init = ["word"]
    attr_init_default = OrderedDict(dict())
    def check_valid(self):
        pass

    @staticmethod
    def fit_func(x, **kwargs):
        return {"cnt": x.shape[0]}

    @staticmethod
    def fit_type(x):
        return x
        #return x.astype(str)

    def fit_cond(self, x):
        """
        数列落在区间 True False
        """
        return (x == self.word)

    def iszero(self):
        return self.cnt == 0

class binstrb(binstrv):
    fit_func = staticmethod(intb.fit_func)

class binstrc(binstrv):
    fit_func = staticmethod(intc.fit_func)

class binList(list):
    """
    区间列
    """
    types = []
    def __getattr__(self, name):
        _info = self.info
        if name in self.info.columns:
            return _info[name]
        else:
            return super().__getattr__(name)
    def __init__(self, i):
        for k in self.types:
            if isinstance(i[0], k):
                self.mode = k.__name__
                break
        super(binList, self).__init__(i)
        self.check_valid()

    def check_valid(self):
        pass

    @property
    def dev_cnt(self):
        _d = dict()
        for j, i in enumerate(self):
            if i.is_stick():
                _d[j] = math.inf
            else:
                _d[j] = i.cnt
        return pd.Series(_d)

    @staticmethod
    def cut_ticks(**kwargs):
        """
        normal entrance 1
        """
        pass

    @property
    def info(self):
        return pd.DataFrame({j: i.info for j, i in enumerate(self)}).T


    def fit(self, *args, **kwargs):
        for i in self:
            i.fit(*args, **kwargs)


    def trans_single(self, x, keyword, default = None):
        assert isinstance(keyword, str)
        x1 = x.copy()
        if default is not None:
            x1.loc[:] = default
        for i in self:
            x1.loc[i.fit_cond(x)] = getattr(i, keyword)
        return x1

    def trans(self, x, keyword, default = None):
        if isinstance(keyword, str):
            return self.trans_single(x, keyword, default = default)
        elif isinstance(keyword, list):
            _df = pd.concat([self.trans_single(x, i, default = default) \
                             for i in keyword], axis = 1)
            _df.columns = keyword
            return _df
        else:
            raise Exception("arg - keyword not valid")

    def calculate(self, func = None, **kwargs):
        """
        update: does result update to info
        """
        if func is None:
            res_df = pd.DataFrame({i.word: i.info for i in self}).T
        else:
            res_df = pd.DataFrame({i.word: i.calculate(func = func, **kwargs) for i in self}).T
        return res_df

    def save_format(self):
        res = {"info":self.info, "mode": self.mode}

        try:
            _name = getattr(self, "name")
        except:
            _name = "_noName"
        res["name"] = _name

        try:
            _res = getattr(self, "result")
            res["result"] = _res
        except:
            pass

        return res

    def save_format_json(self):
        res = self.save_format()
        res["info"] = res["info"].  to_json()
        res["result"] = res["result"].  to_json()
        return json.dumps(res)

    @staticmethod
    def load_format_json(res):
        res = json.loads(res)
        res["info"] = pd.DataFrame(json.loads(res["info"]))
        res["result"] = pd.DataFrame(json.loads(res["result"]))
        return self.load_format(res)

    def save(self, path):
        _data = self.save_format()
        with open(path, "wb") as f:
            pickle.dump(_data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return binList.load_format(_d)

    @staticmethod
    def load_format(_d):
        info = _d["info"]
        mode = _d["mode"]
        name = _d["name"]

        if mode[:3] == "int":
            _m1 = intList
        elif mode[:6] == "binstr":
            _m1 = strList
        else:
            raise Exception("mode not int or str")

        _m = eval(mode)
        _l = list()
        for i, j in info.iterrows():
            _l.append(_m(**j.to_dict()))
        _res = _m1(_l)
        _res.name = name

        ## TODO more gentle
        try:
            _res.result = _d["result"]
        except:
            pass
        return _res

    def reset_info(self):
        for i in self:
            i.reset_info()

class intList(binList):
    """
    区间列
    """
    types = [intb, intc, intv]
    def check_valid(self):
        for i in range(len(self) - 1):
            assert self[i]. rightExact(self[i + 1])
            assert type(self[i]) == type(self[i + 1])
        #assert self.info.isnull().sum().sum() == 0

    @staticmethod
    def fit_type(x):
        return x.astype(float)

    @staticmethod
    def cut_ticks(l, mode = "intv", stable_points = [], single_tick = True, **kwargs):
        """
        normal entrance 1
        """
        mode = "int" + mode[ - 1]. lower()
        if single_tick is False:
            if len(l) == 1:
                l = [l[0], l[0]]
        _l = list()
        _m = eval(mode)
        for i in range(len(l)):
            if single_tick is True:
                _o = _m(left = l[i], right = l[i],
                        lt = "close",
                        rt = "close",
                )
                if l[i] in stable_points:
                    _o.stick()
                _l.append(_o)
                if i <= len(l) - 2:
                    _l.append(_m(left = l[i], right = l[i + 1], lt = "open", rt = "open"))
            else:
                if i == (len(l) - 1):
                    continue
                if i == 0:
                    _lt = "close"
                else:
                    _lt = "open"
                _l.append(_m(left = l[i], right = l[i + 1], lt = _lt, rt = "close"))
        return intList(_l)

    @staticmethod
    def cut_ticks_sample(x, quant, mode = "v", stable_points = [], single_tick = True, **kwargs):
        if stable_points:
            assert single_tick == True
        mode = "int" + mode[ - 1]. lower()
        if not isinstance(quant, list):
            quant = [i / quant for i in range(quant + 1)]
        l1 = pd.Series(x).quantile(quant, interpolation = "nearest").drop_duplicates().tolist()
        if stable_points is not None:
            l1 = pd.Series(l1 + stable_points).drop_duplicates().sort_values().tolist()

        return intList.cut_ticks(l1, mode = mode, single_tick = single_tick, stable_points = stable_points)

    def delim_zero(self):
        _d1 = pd.Series([i.iszero() for i in self])
        _d2 = _d1[_d1 == True]. index.tolist()
        _d2.sort(reverse = True)
        for i in _d2:
            if i > 0:
                self[i - 1]. merge(self[i])
                self.pop(i)
            else:
                self[i + 1]. merge(self[i])
                self.pop(i)

    def cover_inf(self):
        for j, i in enumerate(self):
            _d = dict()
            if j == 0:
                _d["left"] = -math.inf
            if j == (len(self) - 1):
                _d["right"] = math.inf
            i.update(_d)

    def st(self, x, cover_inf = False, **kwargs):
        self.delim_zero()
        for j, i in enumerate(self):
            _d = dict()
            if j == 0:
                if cover_inf is True:
                    _d["left"] = - math.inf
                    _d["lt"] = "open"
                else:
                    _d["left"] = i.fit_df(x, x).min()
                    _d["lt"] = "close"
            else:
                _d["left"] = self[j - 1]. right
                _d["lt"] = "open"

            if j == (len(self) - 1):
                if cover_inf is True:
                    _d["right"] = math.inf
                else:
                    _d["right"] = i.fit_df(x, x).max()
            else:
                _d["right"] = i.fit_df(x, x).max()
            _d["rt"] = "close"
            i.update(_d)

    def ticks_boundaries(self):
        _l = self.left. tolist()
        _l += [math.inf]
        _l[0] = -math.inf
        return _l


    def find_merge_binV(self, ruleV = 500):
        _C = self.dev_cnt
        _C2 = _C[_C < ruleV]
        if not (_C2.shape[0] == 0):
            min_ind = _C2[_C2 == _C2.min()]. index[0]
            return self.merge_target_fill(merge = min_ind, target = None)
        else:
            return {"merge": None, "target": None}

    def merge_target_fill(self, merge, target = None):
        if (merge is not None) and (target is None):
            if merge == 0:
                target = merge + 1
            elif merge == (len(self) - 1):
                target = merge - 1
            else:
                a = self[merge]
                l = self[merge - 1]
                r = self[merge + 1]
                l1 = a.dev(l)
                r1 = a.dev(r)
                if l1 > r1:
                    target = merge + 1
                else:
                    target = merge - 1
        return {"merge": merge, "target": target}

    def find_merge_bin_common(self, rule = 5):
        _C = pd.Series({i:j.dev_next for i, j in enumerate(self[: -1])})
        _C2 = _C[_C < rule]
        if not (_C2.shape[0] == 0):
            min_ind = _C2[_C2 == _C2.min()]. index[0]
            return {"merge": min_ind, "target": min_ind + 1}
        else:
            return {"merge": None, "target": None}

    def find_merge_binB(self, ruleB = 5):
        return self.find_merge_bin_common(rule = ruleB)

    def find_merge_binC(self, ruleC = 0.00001):
        return self.find_merge_bin_common(rule = ruleC)

    def join(self, merge, target):
        self[target]. merge(self[merge])
        _obj = self[target]
        self.pop(merge)
        _obj_ind = self.index(_obj)
        if _obj_ind > 0:
            self.add_dev_next(_obj_ind - 1)
        if _obj_ind < (len(self) - 1):
            self.add_dev_next(_obj_ind)

    def pos_value(self, i, alpha = 1):
        s = 1
        l = len(self)
        if i >= (l - 1):
            return 1
        pos = (self[i]. mean > self[i + 1]. mean)

        if i <= 0:
            s += alpha
        elif self[i - 1]. is_stick():
            s += alpha
        else:
            pos1 = (self[i - 1]. mean > self[i]. mean)
            if pos == pos1:
                s += alpha


        if i >= (l - 2):
            s += alpha
        elif self[i + 2]. is_stick():
            s += alpha
        else:
            pos1 = (self[i + 1]. mean > self[i + 2]. mean)
            if pos == pos1:
                s += alpha
        return s

    def add_dev_next(self, i):
        try:
            pos = self.pos_value(i)
        except:
            #print("no pos info")
            pos = 1
        self[i]. dev_next = self[i]. dev(self[i + 1]) * pos

    def init_dev_next(self):
        for i in range(len(self) - 1):
            self.add_dev_next(i)

    def merge_common(self, rule, ruleM = 1, mode = "V"):
        if mode in ["B", "C"]:
            self.init_dev_next()
        while True:
            if len(self) <= ruleM:
                break
            if mode == "V":
                _d = self.find_merge_binV(ruleV = rule)
            if mode == "B":
                _d = self.find_merge_binB(ruleB = rule)
            if mode == "C":
                _d = self.find_merge_binC(ruleC = rule)
            target = _d["target"]
            merge = _d["merge"]
            if merge is None:
                break
            self.join(merge, target)

    def mergeV(self, ruleV = 500, ruleM = 1, **kwargs):
        self.merge_common(rule = ruleV, ruleM = ruleM, mode = "V")

    def mergeB(self, ruleV = 500, ruleB = 5, ruleM = 1, **kwargs):
        self.merge_common(rule = ruleV, ruleM = ruleM, mode = "V")
        self.merge_common(rule = ruleB, ruleM = ruleM, mode = "B")

    def mergeC(self, ruleV = 500, ruleC = 5, ruleM = 1, **kwargs):
        self.merge_common(rule = ruleV, ruleM = ruleM, mode = "V")
        self.merge_common(rule = ruleC, ruleM = ruleM, mode = "C")

    def merge(self, *kw1, **kwargs):
        if isinstance(self[0], intb):
            print("MERGE B")
            self.mergeB(*kw1, **kwargs)
        elif isinstance(self[0], intc):
            print("MERGE C")
            self.mergeC(*kw1, **kwargs)
        elif isinstance(self[0], intv):
            print("MERGE V")
            self.mergeV(*kw1, **kwargs)
        self.st(*kw1, **kwargs)

class strList(binList):
    """
    区间列
    """
    types = [binstrb, binstrc, binstrv]
    def check_valid(self):
        for i in self:
            pass
            #assert isinstance(i.word, str)

    @staticmethod
    def fit_type(x):
        return x
        #return x.astype(str)


    @staticmethod
    def cut_ticks(l, mode = "binstrv", **kwargs):
        """
        normal entrance 1
        """
        mode = "binstr" + mode[ - 1]. lower()
        _l = list()
        _m = eval(mode)
        _ = [_l.append(_m(word = i)) for i in l]
        return strList(_l)

    @staticmethod
    def cut_ticks_sample(x, mode = "binstrv", **kwargs):
        mode = "binstr" + mode[ - 1]. lower()
        l1 = list(set(x))
        l1.sort()
        if len(l1) > 30:
            raise Exception("too many categories")
        return strList.cut_ticks(l1, mode = mode)

class binning(binList):
    """
    分箱工具
    根据X自动识别分箱类型(字符型 or 区间型)
    fit 不被func所影响
    binning 被func所影响
    """
    @staticmethod
    def guess_type(l):
        def to_npnd(l):
            if isinstance(l, pd.Series):
                return l
            elif isinstance(l, list):
                return pd.Series(l)
            elif (not isinstance(l, np.ndarray)):
                raise Exception("input tick sample error")
            return l
        l = to_npnd(l)
        try:
            l = l.astype(np.float)
            return "int", l
        except:
            return "str", l

    @staticmethod
    def tick(**kwargs):
        """
        确定分箱刻度,
        入参：
        x: 必填
        y: 选填 模式为b或c时必须
        mode: 选填 默认b
        single_tick: 选填 01型
        cover_inf: 选填 01型 是否把区间上下界置为inf
        l: 选填 确定刻度
        返回值:
        model: 箱列对象 包含区间刻度 未包含分箱输出信息

        """
        _type, _x = binning.guess_type(kwargs["x"])
        kwargs["x"] = _x
        model_type = eval(_type + "List")
        if "l" not in kwargs:
            model = model_type.cut_ticks_sample(**kwargs)
            model.fit(**kwargs)
            if _type == "int":
                model.merge(**kwargs)
            return model
        else:
            kwargs["single_tick"] = False
            model = model_type.cut_ticks(**kwargs)
            model.fit(**kwargs)
            return model

    @staticmethod
    def binning(**kwargs):
        """
        入参：
        继承tick的入参另外
        func：根据func来确定分箱输出

        返回：
        model 将输出结果保存在model.result
        """
        model = binning.tick(**kwargs)
        model.result = model.calculate(**kwargs)
        return model

class intLDict(dict):
    """
    intLDict
    用来保存一组分箱的结果
    以文件的形式进行保存和读取
    """
    @staticmethod
    def load_format(_d):
        return intLDict({i: binList.load_format(j) for i, j in _d.items()})

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return intLDict.load_format(_d)

    def save_format(self):
        _data = {i: {** j.save_format(), "name": i} for i, j in self.items()}
        return _data

    def save(self, path):
        _data = self.save_format()
        with open(path, "wb") as f:
            pickle.dump(_data, f)

    def trans(self, X, keyword = "woe", cols = None, default = None):
        _df = pd.DataFrame(index = X.index)
        if cols is None:
            cols = self.keys()
        for i in cols:
            _df[i] = self[i]. trans(X[i], keyword = keyword, default = default)
        return _df

    def cover_inf(self):
        for i, j in self.items():
            try:
                j.cover_inf()
            except:
                print(i, "cannot apply cover_inf!")

    def sub(self, cols):
        return intLDict({i: j for i, j in self.items() if i in cols})

raw_kwargs = {
    "func": lambda self, x, y:{"cnt": x.shape[0], "mean": y.mean()},
    "mode": "b",
    "ruleV": 500,
    "ruleB": 6,
    "ruleM": 1,
    "ruleC": -0.0001,
    "quant": 10,
}

def pd_binning(self, x, y, **kwargs):
    _kwargs = raw_kwargs.copy()
    _kwargs.update(kwargs)
    x = self[x]
    y = self[y]
    model = binning.binning(df = self, x = x, y = y, **_kwargs)
    res = model.result
    res.model = model
    return res

pd.DataFrame.binning = pd_binning

if __name__ == "GG":
    #import VRB.save_pboc2_concat
    #reload(VRB.save_pboc2_concat)
    from VRB.save_pboc2_concat import concat
    x_raw = concat
    x_raw = read_vrb("x_XSMY_2020-12-01_2020-12-06.pkl")
    x_raw["label"] = (x_raw["derived_pbc_sum_l6m_cc_avg_amt_and_l24m_asfbal_pl"]. replace("", -1).astype(float)) > 100000
    x1 = x_raw[["native_uncancellation_list_pjsyed", "label"]]
    x1["score"] = x1["native_uncancellation_list_pjsyed"]. replace("", -1).astype(float)
    x1["score1"] = x1["score"]. apply(lambda x:"asdf" if str(x)[0] in ["3", "4", "5"] else "asdfasd")
    _d = dict()

    i = "score"
    for i in ["score", "score1"]:
        a = binning.binning(df = x1,
                  #x = "score",
                  x = x1[i],
                  y = x1["label"],
                  mode = "b",
                  ruleV = 500,
                  ruleB = x1.shape[0] / 10000,
                  ruleM = 1,
                  ruleC = -0.0001,
                  quant = 10,
        )

    _d1 = intLDict(_d)
    _d1.save_format()
    _d1.save("asdf.pkl")
    _d2 = intLDict.load("asdf.pkl")


    _d2["score1"].trans(x = x1, keyword = ["woe", "mean"])
    _d2["score1"].trans(x = x1, keyword = "woe")
    def kw_to_dict(**kwargs):
        return kwargs


    kwargs = kw_to_dict(
        df = x1,
        x = x1[i],
        func = lambda i, x, y, df:{"cnt": i.cnt, "mean": i.bad_cnt / i.cnt,
                              "cnt1": x.shape[0], "mean1": y.mean},
        y = x1["label"],
        mode = "b",
        ruleV = 500,
        ruleB = 6 / 10000,
        ruleM = 1,
        ruleC = -0.0001,
        quant = 10,
    )


    z.binning(x = "x", y = "label", mode = "v", quant = 10, ruleV = z.shape[0] / 20)
    z.binning(x = "x", y = "label", mode = "v", quant = 10, ruleV = z.shape[0] / 20,
              l = [0, 0.5, 1], single_tick = False
    )
    import sqlqueryV2 as sqq
    log2 = lgt(X = x_raw.iloc[:, :30], Y = x_raw["label"])
    log2.binning()
    cond1 = log2.X["pid"] > "PID20201204"
    cond2 = log2.X["pid"] <= "PID20201204"
    conds = [cond1, cond2]
    #$subs = log2.draw_binning(conds = conds)
    #$log2.draw_binning_excel()
    log2.update_woe()
    cols = log2.var().index.tolist()
    res = log2.train(cols = cols, sample = conds)

    res["result"]["set1"]
    res["result"]["set2"]
