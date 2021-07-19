var echartsExample;
echartsExample = (() => {
    var t = {
            913: (t, e, n) => {
                t = n.nmd(t);
                var a = {
                        grid: "GridComponent",
                        polar: "PolarComponent",
                        geo: "GeoComponent",
                        singleAxis: "SingleAxisComponent",
                        parallel: "ParallelComponent",
                        calendar: "CalendarComponent",
                        graphic: "GraphicComponent",
                        toolbox: "ToolboxComponent",
                        tooltip: "TooltipComponent",
                        axisPointer: "AxisPointerComponent",
                        brush: "BrushComponent",
                        title: "TitleComponent",
                        timeline: "TimelineComponent",
                        markPoint: "MarkPointComponent",
                        markLine: "MarkLineComponent",
                        markArea: "MarkAreaComponent",
                        legend: "LegendComponent",
                        dataZoom: "DataZoomComponent",
                        visualMap: "VisualMapComponent",
                        aria: "AriaComponent",
                        dataset: "DatasetComponent",
                        xAxis: "GridComponent",
                        yAxis: "GridComponent",
                        angleAxis: "PolarComponent",
                        radiusAxis: "PolarComponent"
                    },
                    i = {
                        line: "LineChart",
                        bar: "BarChart",
                        pie: "PieChart",
                        scatter: "ScatterChart",
                        radar: "RadarChart",
                        map: "MapChart",
                        tree: "TreeChart",
                        treemap: "TreemapChart",
                        graph: "GraphChart",
                        gauge: "GaugeChart",
                        funnel: "FunnelChart",
                        parallel: "ParallelChart",
                        sankey: "SankeyChart",
                        boxplot: "BoxplotChart",
                        candlestick: "CandlestickChart",
                        effectScatter: "EffectScatterChart",
                        lines: "LinesChart",
                        heatmap: "HeatmapChart",
                        pictorialBar: "PictorialBarChart",
                        themeRiver: "ThemeRiverChart",
                        sunburst: "SunburstChart",
                        custom: "CustomChart"
                    },
                    o = {
                        grid3D: "Grid3DComponent",
                        geo3D: "Geo3DComponent",
                        globe: "GlobeComponent",
                        mapbox3D: "Mapbox3DComponent",
                        maptalks3D: "Maptalks3DComponent",
                        xAxis3D: "Grid3DComponent",
                        yAxis3D: "Grid3DComponent",
                        zAxis3D: "Grid3DComponent"
                    },
                    r = {
                        bar3D: "Bar3DChart",
                        line3D: "Line3DChart",
                        scatter3D: "Scatter3DChart",
                        lines3D: "Lines3DChart",
                        polygons3D: "Polygons3DChart",
                        surface: "SurfaceChart",
                        map3D: "Map3DChart",
                        scatterGL: "ScatterGLChart",
                        graphGL: "GraphGLChart",
                        flowGL: "FlowGLChart",
                        linesGL: "LinesGLChart"
                    },
                    l = {},
                    s = {},
                    c = {},
                    u = {},
                    d = {
                        SVGRenderer: "svg",
                        CanvasRenderer: "canvas"
                    },
                    p = ["markLine", "markArea", "markPoint"],
                    f = [].concat(p, ["grid", "axisPointer", "aria"]),
                    g = ["xAxis", "yAxis", "angleAxis", "radiusAxis", "xAxis3D", "yAxis3D", "zAxis3D"];

                function m(t, e) {
                    Object.keys(t).forEach((function(n) {
                        g.includes(n) || (e[t[n]] = n)
                    }))
                }

                function h(t, e) {
                    var n = [],
                        a = [],
                        i = [],
                        o = [],
                        r = [];
                    t.forEach((function(t) {
                        t.endsWith("Renderer") ? r.push(t) : s[t] ? (n.push(t), e && n.push(t.replace(/Chart$/, "SeriesOption"))) : l[t] ? (a.push(t), e && a.push(t.replace(/Component$/, "ComponentOption"))) : c[t] ? i.push(t) : u[t] && o.push(t)
                    }));
                    var d = [].concat(a, n, o, i, r),
                        p = "\ntype ECOption = echarts.ComposeOption<\n    ".concat(d.filter((function(t) {
                            return t.endsWith("Option")
                        })).join(" | "), "\n>"),
                        f = [
                            [a, "echarts/components"],
                            [n, "echarts/charts"],
                            [r, "echarts/renderers"],
                            [i, "echarts-gl/charts"],
                            [o, "echarts-gl/components"]
                        ].filter((function(t) {
                            return t[0].length > 0
                        })).map((function(t) {
                            return "\nimport {".concat((e = t[0], "".concat(e.map((function(t) {
                                return "\n    ".concat(t)
                            })).join(","))), "\n} from '").concat(t[1], "';\n    ").trim();
                            var e
                        })).join("\n");
                    return "import * as echarts from 'echarts/core';\n".concat(f, "\n\necharts.use(\n    [").concat(d.filter((function(t) {
                        return !t.endsWith("Option")
                    })).join(", "), "]\n);\n") + (e ? p : "")
                }

                function C(t, e) {
                    var n = [];
                    return t.forEach((function(t) {
                        t.endsWith("Renderer") && "CanvasRenderer" !== t ? n.push("zrender/lib/".concat(d[t], "/").concat(d[t])) : s[t] ? n.push("echarts/lib/chart/".concat(s[t])) : l[t] ? n.push("echarts/lib/component/".concat(l[t])) : c[t] ? n.push("echarts-gl/lib/chart/".concat(c[t])) : u[t] && n.push("echarts-gl/lib/component/".concat(u[t]))
                    })), e ? "import * as echarts from 'echarts/lib/echarts';\n".concat(n.map((function(t) {
                        return "import '".concat(t, "';")
                    })).join("\n"), "\n") : "const echarts = require('echarts/lib/echarts');\n".concat(n.map((function(t) {
                        return "require('".concat(t, "');")
                    })).join("\n"), "\n")
                }

                function y(t) {
                    return !!t.find((function(t) {
                        return !(!c[t] && !u[t])
                    }))
                }
                m(a, l), m(i, s), m(o, u), m(r, c), t.exports.collectDeps = function t(e) {
                    var n = [];
                    if (e.options) return e.options.forEach((function(e) {
                        n = n.concat(t(e))
                    })), e.baseOption && (n = n.concat(t(e.baseOption))), Array.from(new Set(n));
                    Object.keys(e).forEach((function(t) {
                        if (!f.includes(t)) {
                            var i = e[t];
                            Array.isArray(i) && !i.length || (a[t] && n.push(a[t]), o[t] && n.push(o[t]))
                        }
                    }));
                    var l = e.series;
                    return Array.isArray(l) || (l = [l]), l.forEach((function(t) {
                        i[t.type] && n.push(i[t.type]), r[t.type] && n.push(r[t.type]), "map" === t.type && n.push(a.geo), p.forEach((function(e) {
                            t[e] && n.push(a[e])
                        }))
                    })), Array.from(new Set(n))
                }, t.exports.buildMinimalBundleCode = h, t.buildLegacyMinimalBundleCode = C, t.exports.buildExampleCode = function(t, e, n) {
                    var a = n.minimal,
                        i = n.esm,
                        o = void 0 === i || i,
                        r = n.legacy,
                        l = n.ts,
                        s = n.theme,
                        c = n.ROOT_PATH,
                        u = n.extraImports;
                    l && (o = !0), a && !o && (r = !0);
                    var d = t.indexOf("ecStat") >= 0,
                        p = t.indexOf("ROOT_PATH") >= 0,
                        f = t.indexOf("app") >= 0,
                        g = "\n".concat(d ? o ? "import ecStat from 'echarts-stat';" : "var ecStat = require('echarts-stat');" : "", "\n"),
                        m = [a ? r ? C(e, o) : h(e, l) : o ? "import * as echarts from 'echarts';".concat(y(e) ? "\nimport 'echarts-gl';" : "") : "var echarts = require('echarts');".concat(y(e) ? "\nrequire('echarts-gl');" : ""), s && "dark" !== s ? o ? "import 'echarts/theme/".concat(s, "'") : "require('echarts/theme/".concat(s, "')") : "", u].filter((function(t) {
                            return !!t
                        })).join("\n"),
                        v = [p ? "var ROOT_PATH = '".concat(c, "';") : "", f ? "var app".concat(l ? ": any" : "", " = {};") : "", l && !a ? "type ECOption = echarts.EChartsOption" : ""].filter((function(t) {
                            return !!t
                        })).join("\n"),
                        b = [m.trim(), g.trim(), v.trim()].filter((function(t) {
                            return !!t
                        })).join("\n\n");
                    return "".concat(b, "\n\nvar chartDom = document.getElementById('main')").concat(l ? "!" : "", ";\nvar myChart = echarts.init(chartDom").concat(s ? ", '".concat(s, "'") : "", ");\nvar option").concat(l ? ": ECOption" : "", ";\n\n").concat(t.trim(), "\n\noption && myChart.setOption(option);\n")
                }
            },
            403: (t, e, n) => {
                "use strict";
                n.r(e), n.d(e, {
                    init: () => Sl
                });
                const a = Vue;
                var i = n.n(a),
                    o = ["style", "currency", "currencyDisplay", "useGrouping", "minimumIntegerDigits", "minimumFractionDigits", "maximumFractionDigits", "minimumSignificantDigits", "maximumSignificantDigits", "localeMatcher", "formatMatcher", "unit"];

                function r(t, e) {
                    "undefined" != typeof console && (console.warn("[vue-i18n] " + t), e && console.warn(e.stack))
                }
                var l = Array.isArray;

                function s(t) {
                    return null !== t && "object" == typeof t
                }

                function c(t) {
                    return "string" == typeof t
                }
                var u = Object.prototype.toString;

                function d(t) {
                    return "[object Object]" === u.call(t)
                }

                function p(t) {
                    return null == t
                }

                function f() {
                    for (var t = [], e = arguments.length; e--;) t[e] = arguments[e];
                    var n = null,
                        a = null;
                    return 1 === t.length ? s(t[0]) || Array.isArray(t[0]) ? a = t[0] : "string" == typeof t[0] && (n = t[0]) : 2 === t.length && ("string" == typeof t[0] && (n = t[0]), (s(t[1]) || Array.isArray(t[1])) && (a = t[1])), {
                        locale: n,
                        params: a
                    }
                }

                function g(t) {
                    return JSON.parse(JSON.stringify(t))
                }

                function m(t, e) {
                    return !!~t.indexOf(e)
                }
                var h = Object.prototype.hasOwnProperty;

                function C(t, e) {
                    return h.call(t, e)
                }

                function y(t) {
                    for (var e = arguments, n = Object(t), a = 1; a < arguments.length; a++) {
                        var i = e[a];
                        if (null != i) {
                            var o = void 0;
                            for (o in i) C(i, o) && (s(i[o]) ? n[o] = y(n[o], i[o]) : n[o] = i[o])
                        }
                    }
                    return n
                }

                function v(t, e) {
                    if (t === e) return !0;
                    var n = s(t),
                        a = s(e);
                    if (!n || !a) return !n && !a && String(t) === String(e);
                    try {
                        var i = Array.isArray(t),
                            o = Array.isArray(e);
                        if (i && o) return t.length === e.length && t.every((function(t, n) {
                            return v(t, e[n])
                        }));
                        if (i || o) return !1;
                        var r = Object.keys(t),
                            l = Object.keys(e);
                        return r.length === l.length && r.every((function(n) {
                            return v(t[n], e[n])
                        }))
                    } catch (t) {
                        return !1
                    }
                }
                var b = {
                        beforeCreate: function() {
                            var t = this.$options;
                            if (t.i18n = t.i18n || (t.__i18n ? {} : null), t.i18n) {
                                if (t.i18n instanceof K) {
                                    if (t.__i18n) try {
                                        var e = {};
                                        t.__i18n.forEach((function(t) {
                                            e = y(e, JSON.parse(t))
                                        })), Object.keys(e).forEach((function(n) {
                                            t.i18n.mergeLocaleMessage(n, e[n])
                                        }))
                                    } catch (t) {}
                                    this._i18n = t.i18n, this._i18nWatcher = this._i18n.watchI18nData()
                                } else if (d(t.i18n)) {
                                    var n = this.$root && this.$root.$i18n && this.$root.$i18n instanceof K ? this.$root.$i18n : null;
                                    if (n && (t.i18n.root = this.$root, t.i18n.formatter = n.formatter, t.i18n.fallbackLocale = n.fallbackLocale, t.i18n.formatFallbackMessages = n.formatFallbackMessages, t.i18n.silentTranslationWarn = n.silentTranslationWarn, t.i18n.silentFallbackWarn = n.silentFallbackWarn, t.i18n.pluralizationRules = n.pluralizationRules, t.i18n.preserveDirectiveContent = n.preserveDirectiveContent), t.__i18n) try {
                                        var a = {};
                                        t.__i18n.forEach((function(t) {
                                            a = y(a, JSON.parse(t))
                                        })), t.i18n.messages = a
                                    } catch (t) {}
                                    var i = t.i18n.sharedMessages;
                                    i && d(i) && (t.i18n.messages = y(t.i18n.messages, i)), this._i18n = new K(t.i18n), this._i18nWatcher = this._i18n.watchI18nData(), (void 0 === t.i18n.sync || t.i18n.sync) && (this._localeWatcher = this.$i18n.watchLocale()), n && n.onComponentInstanceCreated(this._i18n)
                                }
                            } else this.$root && this.$root.$i18n && this.$root.$i18n instanceof K ? this._i18n = this.$root.$i18n : t.parent && t.parent.$i18n && t.parent.$i18n instanceof K && (this._i18n = t.parent.$i18n)
                        },
                        beforeMount: function() {
                            var t = this.$options;
                            t.i18n = t.i18n || (t.__i18n ? {} : null), t.i18n ? (t.i18n instanceof K || d(t.i18n)) && (this._i18n.subscribeDataChanging(this), this._subscribing = !0) : (this.$root && this.$root.$i18n && this.$root.$i18n instanceof K || t.parent && t.parent.$i18n && t.parent.$i18n instanceof K) && (this._i18n.subscribeDataChanging(this), this._subscribing = !0)
                        },
                        beforeDestroy: function() {
                            if (this._i18n) {
                                var t = this;
                                this.$nextTick((function() {
                                    t._subscribing && (t._i18n.unsubscribeDataChanging(t), delete t._subscribing), t._i18nWatcher && (t._i18nWatcher(), t._i18n.destroyVM(), delete t._i18nWatcher), t._localeWatcher && (t._localeWatcher(), delete t._localeWatcher)
                                }))
                            }
                        }
                    },
                    _ = {
                        name: "i18n",
                        functional: !0,
                        props: {
                            tag: {
                                type: [String, Boolean, Object],
                                default: "span"
                            },
                            path: {
                                type: String,
                                required: !0
                            },
                            locale: {
                                type: String
                            },
                            places: {
                                type: [Array, Object]
                            }
                        },
                        render: function(t, e) {
                            var n = e.data,
                                a = e.parent,
                                i = e.props,
                                o = e.slots,
                                r = a.$i18n;
                            if (r) {
                                var l = i.path,
                                    s = i.locale,
                                    c = i.places,
                                    u = o(),
                                    d = r.i(l, s, function(t) {
                                        var e;
                                        for (e in t)
                                            if ("default" !== e) return !1;
                                        return Boolean(e)
                                    }(u) || c ? function(t, e) {
                                        var n = e ? function(t) {
                                            return Array.isArray(t) ? t.reduce(w, {}) : Object.assign({}, t)
                                        }(e) : {};
                                        if (!t) return n;
                                        var a = (t = t.filter((function(t) {
                                            return t.tag || "" !== t.text.trim()
                                        }))).every(x);
                                        return t.reduce(a ? L : w, n)
                                    }(u.default, c) : u),
                                    p = i.tag && !0 !== i.tag || !1 === i.tag ? i.tag : "span";
                                return p ? t(p, n, d) : d
                            }
                        }
                    };

                function L(t, e) {
                    return e.data && e.data.attrs && e.data.attrs.place && (t[e.data.attrs.place] = e), t
                }

                function w(t, e, n) {
                    return t[n] = e, t
                }

                function x(t) {
                    return Boolean(t.data && t.data.attrs && t.data.attrs.place)
                }
                var k, S = {
                    name: "i18n-n",
                    functional: !0,
                    props: {
                        tag: {
                            type: [String, Boolean, Object],
                            default: "span"
                        },
                        value: {
                            type: Number,
                            required: !0
                        },
                        format: {
                            type: [String, Object]
                        },
                        locale: {
                            type: String
                        }
                    },
                    render: function(t, e) {
                        var n = e.props,
                            a = e.parent,
                            i = e.data,
                            r = a.$i18n;
                        if (!r) return null;
                        var l = null,
                            u = null;
                        c(n.format) ? l = n.format : s(n.format) && (n.format.key && (l = n.format.key), u = Object.keys(n.format).reduce((function(t, e) {
                            var a;
                            return m(o, e) ? Object.assign({}, t, ((a = {})[e] = n.format[e], a)) : t
                        }), null));
                        var d = n.locale || r.locale,
                            p = r._ntp(n.value, d, l, u),
                            f = p.map((function(t, e) {
                                var n, a = i.scopedSlots && i.scopedSlots[t.type];
                                return a ? a(((n = {})[t.type] = t.value, n.index = e, n.parts = p, n)) : t.value
                            })),
                            g = n.tag && !0 !== n.tag || !1 === n.tag ? n.tag : "span";
                        return g ? t(g, {
                            attrs: i.attrs,
                            class: i.class,
                            staticClass: i.staticClass
                        }, f) : f
                    }
                };

                function N(t, e, n) {
                    O(0, n) && D(t, e, n)
                }

                function M(t, e, n, a) {
                    if (O(0, n)) {
                        var i = n.context.$i18n;
                        (function(t, e) {
                            var n = e.context;
                            return t._locale === n.$i18n.locale
                        })(t, n) && v(e.value, e.oldValue) && v(t._localeMessage, i.getLocaleMessage(i.locale)) || D(t, e, n)
                    }
                }

                function T(t, e, n, a) {
                    if (n.context) {
                        var i = n.context.$i18n || {};
                        e.modifiers.preserve || i.preserveDirectiveContent || (t.textContent = ""), t._vt = void 0, delete t._vt, t._locale = void 0, delete t._locale, t._localeMessage = void 0, delete t._localeMessage
                    } else r("Vue instance does not exists in VNode context")
                }

                function O(t, e) {
                    var n = e.context;
                    return n ? !!n.$i18n || (r("VueI18n instance does not exists in Vue instance"), !1) : (r("Vue instance does not exists in VNode context"), !1)
                }

                function D(t, e, n) {
                    var a, i, o = function(t) {
                            var e, n, a, i;
                            return c(t) ? e = t : d(t) && (e = t.path, n = t.locale, a = t.args, i = t.choice), {
                                path: e,
                                locale: n,
                                args: a,
                                choice: i
                            }
                        }(e.value),
                        l = o.path,
                        s = o.locale,
                        u = o.args,
                        p = o.choice;
                    if (l || s || u)
                        if (l) {
                            var f = n.context;
                            t._vt = t.textContent = null != p ? (a = f.$i18n).tc.apply(a, [l, p].concat(A(s, u))) : (i = f.$i18n).t.apply(i, [l].concat(A(s, u))), t._locale = f.$i18n.locale, t._localeMessage = f.$i18n.getLocaleMessage(f.$i18n.locale)
                        } else r("`path` is required in v-t directive");
                    else r("value type not supported")
                }

                function A(t, e) {
                    var n = [];
                    return t && n.push(t), e && (Array.isArray(e) || d(e)) && n.push(e), n
                }

                function E(t) {
                    E.installed = !0, (k = t).version && Number(k.version.split(".")[0]),
                        function(t) {
                            t.prototype.hasOwnProperty("$i18n") || Object.defineProperty(t.prototype, "$i18n", {
                                get: function() {
                                    return this._i18n
                                }
                            }), t.prototype.$t = function(t) {
                                for (var e = [], n = arguments.length - 1; n-- > 0;) e[n] = arguments[n + 1];
                                var a = this.$i18n;
                                return a._t.apply(a, [t, a.locale, a._getMessages(), this].concat(e))
                            }, t.prototype.$tc = function(t, e) {
                                for (var n = [], a = arguments.length - 2; a-- > 0;) n[a] = arguments[a + 2];
                                var i = this.$i18n;
                                return i._tc.apply(i, [t, i.locale, i._getMessages(), this, e].concat(n))
                            }, t.prototype.$te = function(t, e) {
                                var n = this.$i18n;
                                return n._te(t, n.locale, n._getMessages(), e)
                            }, t.prototype.$d = function(t) {
                                for (var e, n = [], a = arguments.length - 1; a-- > 0;) n[a] = arguments[a + 1];
                                return (e = this.$i18n).d.apply(e, [t].concat(n))
                            }, t.prototype.$n = function(t) {
                                for (var e, n = [], a = arguments.length - 1; a-- > 0;) n[a] = arguments[a + 1];
                                return (e = this.$i18n).n.apply(e, [t].concat(n))
                            }
                        }(k), k.mixin(b), k.directive("t", {
                            bind: N,
                            update: M,
                            unbind: T
                        }), k.component(_.name, _), k.component(S.name, S), k.config.optionMergeStrategies.i18n = function(t, e) {
                            return void 0 === e ? t : e
                        }
                }
                var P = function() {
                    this._caches = Object.create(null)
                };
                P.prototype.interpolate = function(t, e) {
                    if (!e) return [t];
                    var n = this._caches[t];
                    return n || (n = function(t) {
                            for (var e = [], n = 0, a = ""; n < t.length;) {
                                var i = t[n++];
                                if ("{" === i) {
                                    a && e.push({
                                        type: "text",
                                        value: a
                                    }), a = "";
                                    var o = "";
                                    for (i = t[n++]; void 0 !== i && "}" !== i;) o += i, i = t[n++];
                                    var r = "}" === i,
                                        l = F.test(o) ? "list" : r && R.test(o) ? "named" : "unknown";
                                    e.push({
                                        value: o,
                                        type: l
                                    })
                                } else "%" === i ? "{" !== t[n] && (a += i) : a += i
                            }
                            return a && e.push({
                                type: "text",
                                value: a
                            }), e
                        }(t), this._caches[t] = n),
                        function(t, e) {
                            var n = [],
                                a = 0,
                                i = Array.isArray(e) ? "list" : s(e) ? "named" : "unknown";
                            if ("unknown" === i) return n;
                            for (; a < t.length;) {
                                var o = t[a];
                                switch (o.type) {
                                    case "text":
                                        n.push(o.value);
                                        break;
                                    case "list":
                                        n.push(e[parseInt(o.value, 10)]);
                                        break;
                                    case "named":
                                        "named" === i && n.push(e[o.value])
                                }
                                a++
                            }
                            return n
                        }(n, e)
                };
                var F = /^(?:\d)+/,
                    R = /^(?:\w)+/,
                    I = [];
                I[0] = {
                    ws: [0],
                    ident: [3, 0],
                    "[": [4],
                    eof: [7]
                }, I[1] = {
                    ws: [1],
                    ".": [2],
                    "[": [4],
                    eof: [7]
                }, I[2] = {
                    ws: [2],
                    ident: [3, 0],
                    0: [3, 0],
                    number: [3, 0]
                }, I[3] = {
                    ident: [3, 0],
                    0: [3, 0],
                    number: [3, 0],
                    ws: [1, 1],
                    ".": [2, 1],
                    "[": [4, 1],
                    eof: [7, 1]
                }, I[4] = {
                    "'": [5, 0],
                    '"': [6, 0],
                    "[": [4, 2],
                    "]": [1, 3],
                    eof: 8,
                    else: [4, 0]
                }, I[5] = {
                    "'": [4, 0],
                    eof: 8,
                    else: [5, 0]
                }, I[6] = {
                    '"': [4, 0],
                    eof: 8,
                    else: [6, 0]
                };
                var B = /^\s?(?:true|false|-?[\d.]+|'[^']*'|"[^"]*")\s?$/;

                function j(t) {
                    if (null == t) return "eof";
                    switch (t.charCodeAt(0)) {
                        case 91:
                        case 93:
                        case 46:
                        case 34:
                        case 39:
                            return t;
                        case 95:
                        case 36:
                        case 45:
                            return "ident";
                        case 9:
                        case 10:
                        case 13:
                        case 160:
                        case 65279:
                        case 8232:
                        case 8233:
                            return "ws"
                    }
                    return "ident"
                }
                var G = function() {
                    this._cache = Object.create(null)
                };
                G.prototype.parsePath = function(t) {
                    var e = this._cache[t];
                    return e || (e = function(t) {
                        var e, n, a, i, o, r, l, s = [],
                            c = -1,
                            u = 0,
                            d = 0,
                            p = [];

                        function f() {
                            var e = t[c + 1];
                            if (5 === u && "'" === e || 6 === u && '"' === e) return c++, a = "\\" + e, p[0](), !0
                        }
                        for (p[1] = function() {
                                void 0 !== n && (s.push(n), n = void 0)
                            }, p[0] = function() {
                                void 0 === n ? n = a : n += a
                            }, p[2] = function() {
                                p[0](), d++
                            }, p[3] = function() {
                                if (d > 0) d--, u = 4, p[0]();
                                else {
                                    if (d = 0, void 0 === n) return !1;
                                    if (!1 === (n = function(t) {
                                            var e, n = t.trim();
                                            return ("0" !== t.charAt(0) || !isNaN(t)) && (e = n, B.test(e) ? function(t) {
                                                var e = t.charCodeAt(0);
                                                return e !== t.charCodeAt(t.length - 1) || 34 !== e && 39 !== e ? t : t.slice(1, -1)
                                            }(n) : "*" + n)
                                        }(n))) return !1;
                                    p[1]()
                                }
                            }; null !== u;)
                            if (c++, "\\" !== (e = t[c]) || !f()) {
                                if (i = j(e), 8 === (o = (l = I[u])[i] || l.else || 8)) return;
                                if (u = o[0], (r = p[o[1]]) && (a = void 0 === (a = o[2]) ? e : a, !1 === r())) return;
                                if (7 === u) return s
                            }
                    }(t)) && (this._cache[t] = e), e || []
                }, G.prototype.getPathValue = function(t, e) {
                    if (!s(t)) return null;
                    var n = this.parsePath(e);
                    if (0 === n.length) return null;
                    for (var a = n.length, i = t, o = 0; o < a;) {
                        var r = i[n[o]];
                        if (void 0 === r) return null;
                        i = r, o++
                    }
                    return i
                };
                var z, U = /<\/?[\w\s="/.':;#-\/]+>/,
                    Z = /(?:@(?:\.[a-z]+)?:(?:[\w\-_|.]+|\([\w\-_|.]+\)))/g,
                    V = /^@(?:\.([a-z]+))?:/,
                    W = /[()]/g,
                    H = {
                        upper: function(t) {
                            return t.toLocaleUpperCase()
                        },
                        lower: function(t) {
                            return t.toLocaleLowerCase()
                        },
                        capitalize: function(t) {
                            return "" + t.charAt(0).toLocaleUpperCase() + t.substr(1)
                        }
                    },
                    q = new P,
                    K = function(t) {
                        var e = this;
                        void 0 === t && (t = {}), !k && "undefined" != typeof window && window.Vue && E(window.Vue);
                        var n = t.locale || "en-US",
                            a = !1 !== t.fallbackLocale && (t.fallbackLocale || "en-US"),
                            i = t.messages || {},
                            o = t.dateTimeFormats || {},
                            r = t.numberFormats || {};
                        this._vm = null, this._formatter = t.formatter || q, this._modifiers = t.modifiers || {}, this._missing = t.missing || null, this._root = t.root || null, this._sync = void 0 === t.sync || !!t.sync, this._fallbackRoot = void 0 === t.fallbackRoot || !!t.fallbackRoot, this._formatFallbackMessages = void 0 !== t.formatFallbackMessages && !!t.formatFallbackMessages, this._silentTranslationWarn = void 0 !== t.silentTranslationWarn && t.silentTranslationWarn, this._silentFallbackWarn = void 0 !== t.silentFallbackWarn && !!t.silentFallbackWarn, this._dateTimeFormatters = {}, this._numberFormatters = {}, this._path = new G, this._dataListeners = [], this._componentInstanceCreatedListener = t.componentInstanceCreatedListener || null, this._preserveDirectiveContent = void 0 !== t.preserveDirectiveContent && !!t.preserveDirectiveContent, this.pluralizationRules = t.pluralizationRules || {}, this._warnHtmlInMessage = t.warnHtmlInMessage || "off", this._postTranslation = t.postTranslation || null, this.getChoiceIndex = function(t, n) {
                            var a, i, o = Object.getPrototypeOf(e);
                            return o && o.getChoiceIndex ? o.getChoiceIndex.call(e, t, n) : e.locale in e.pluralizationRules ? e.pluralizationRules[e.locale].apply(e, [t, n]) : (a = t, i = n, a = Math.abs(a), 2 === i ? a ? a > 1 ? 1 : 0 : 1 : a ? Math.min(a, 2) : 0)
                        }, this._exist = function(t, n) {
                            return !(!t || !n || p(e._path.getPathValue(t, n)) && !t[n])
                        }, "warn" !== this._warnHtmlInMessage && "error" !== this._warnHtmlInMessage || Object.keys(i).forEach((function(t) {
                            e._checkLocaleMessage(t, e._warnHtmlInMessage, i[t])
                        })), this._initVM({
                            locale: n,
                            fallbackLocale: a,
                            messages: i,
                            dateTimeFormats: o,
                            numberFormats: r
                        })
                    },
                    J = {
                        vm: {
                            configurable: !0
                        },
                        messages: {
                            configurable: !0
                        },
                        dateTimeFormats: {
                            configurable: !0
                        },
                        numberFormats: {
                            configurable: !0
                        },
                        availableLocales: {
                            configurable: !0
                        },
                        locale: {
                            configurable: !0
                        },
                        fallbackLocale: {
                            configurable: !0
                        },
                        formatFallbackMessages: {
                            configurable: !0
                        },
                        missing: {
                            configurable: !0
                        },
                        formatter: {
                            configurable: !0
                        },
                        silentTranslationWarn: {
                            configurable: !0
                        },
                        silentFallbackWarn: {
                            configurable: !0
                        },
                        preserveDirectiveContent: {
                            configurable: !0
                        },
                        warnHtmlInMessage: {
                            configurable: !0
                        },
                        postTranslation: {
                            configurable: !0
                        }
                    };
                K.prototype._checkLocaleMessage = function(t, e, n) {
                    var a = function(t, e, n, i) {
                        if (d(n)) Object.keys(n).forEach((function(o) {
                            var r = n[o];
                            d(r) ? (i.push(o), i.push("."), a(t, e, r, i), i.pop(), i.pop()) : (i.push(o), a(t, e, r, i), i.pop())
                        }));
                        else if (Array.isArray(n)) n.forEach((function(n, o) {
                            d(n) ? (i.push("[" + o + "]"), i.push("."), a(t, e, n, i), i.pop(), i.pop()) : (i.push("[" + o + "]"), a(t, e, n, i), i.pop())
                        }));
                        else if (c(n) && U.test(n)) {
                            var o = "Detected HTML in message '" + n + "' of keypath '" + i.join("") + "' at '" + e + "'. Consider component interpolation with '<i18n>' to avoid XSS. See https://bit.ly/2ZqJzkp";
                            "warn" === t ? r(o) : "error" === t && function(t, e) {
                                "undefined" != typeof console && console.error("[vue-i18n] " + t)
                            }(o)
                        }
                    };
                    a(e, t, n, [])
                }, K.prototype._initVM = function(t) {
                    var e = k.config.silent;
                    k.config.silent = !0, this._vm = new k({
                        data: t
                    }), k.config.silent = e
                }, K.prototype.destroyVM = function() {
                    this._vm.$destroy()
                }, K.prototype.subscribeDataChanging = function(t) {
                    this._dataListeners.push(t)
                }, K.prototype.unsubscribeDataChanging = function(t) {
                    ! function(t, e) {
                        if (t.length) {
                            var n = t.indexOf(e);
                            n > -1 && t.splice(n, 1)
                        }
                    }(this._dataListeners, t)
                }, K.prototype.watchI18nData = function() {
                    var t = this;
                    return this._vm.$watch("$data", (function() {
                        for (var e = t._dataListeners.length; e--;) k.nextTick((function() {
                            t._dataListeners[e] && t._dataListeners[e].$forceUpdate()
                        }))
                    }), {
                        deep: !0
                    })
                }, K.prototype.watchLocale = function() {
                    if (!this._sync || !this._root) return null;
                    var t = this._vm;
                    return this._root.$i18n.vm.$watch("locale", (function(e) {
                        t.$set(t, "locale", e), t.$forceUpdate()
                    }), {
                        immediate: !0
                    })
                }, K.prototype.onComponentInstanceCreated = function(t) {
                    this._componentInstanceCreatedListener && this._componentInstanceCreatedListener(t, this)
                }, J.vm.get = function() {
                    return this._vm
                }, J.messages.get = function() {
                    return g(this._getMessages())
                }, J.dateTimeFormats.get = function() {
                    return g(this._getDateTimeFormats())
                }, J.numberFormats.get = function() {
                    return g(this._getNumberFormats())
                }, J.availableLocales.get = function() {
                    return Object.keys(this.messages).sort()
                }, J.locale.get = function() {
                    return this._vm.locale
                }, J.locale.set = function(t) {
                    this._vm.$set(this._vm, "locale", t)
                }, J.fallbackLocale.get = function() {
                    return this._vm.fallbackLocale
                }, J.fallbackLocale.set = function(t) {
                    this._localeChainCache = {}, this._vm.$set(this._vm, "fallbackLocale", t)
                }, J.formatFallbackMessages.get = function() {
                    return this._formatFallbackMessages
                }, J.formatFallbackMessages.set = function(t) {
                    this._formatFallbackMessages = t
                }, J.missing.get = function() {
                    return this._missing
                }, J.missing.set = function(t) {
                    this._missing = t
                }, J.formatter.get = function() {
                    return this._formatter
                }, J.formatter.set = function(t) {
                    this._formatter = t
                }, J.silentTranslationWarn.get = function() {
                    return this._silentTranslationWarn
                }, J.silentTranslationWarn.set = function(t) {
                    this._silentTranslationWarn = t
                }, J.silentFallbackWarn.get = function() {
                    return this._silentFallbackWarn
                }, J.silentFallbackWarn.set = function(t) {
                    this._silentFallbackWarn = t
                }, J.preserveDirectiveContent.get = function() {
                    return this._preserveDirectiveContent
                }, J.preserveDirectiveContent.set = function(t) {
                    this._preserveDirectiveContent = t
                }, J.warnHtmlInMessage.get = function() {
                    return this._warnHtmlInMessage
                }, J.warnHtmlInMessage.set = function(t) {
                    var e = this,
                        n = this._warnHtmlInMessage;
                    if (this._warnHtmlInMessage = t, n !== t && ("warn" === t || "error" === t)) {
                        var a = this._getMessages();
                        Object.keys(a).forEach((function(t) {
                            e._checkLocaleMessage(t, e._warnHtmlInMessage, a[t])
                        }))
                    }
                }, J.postTranslation.get = function() {
                    return this._postTranslation
                }, J.postTranslation.set = function(t) {
                    this._postTranslation = t
                }, K.prototype._getMessages = function() {
                    return this._vm.messages
                }, K.prototype._getDateTimeFormats = function() {
                    return this._vm.dateTimeFormats
                }, K.prototype._getNumberFormats = function() {
                    return this._vm.numberFormats
                }, K.prototype._warnDefault = function(t, e, n, a, i, o) {
                    if (!p(n)) return n;
                    if (this._missing) {
                        var r = this._missing.apply(null, [t, e, a, i]);
                        if (c(r)) return r
                    }
                    if (this._formatFallbackMessages) {
                        var l = f.apply(void 0, i);
                        return this._render(e, o, l.params, e)
                    }
                    return e
                }, K.prototype._isFallbackRoot = function(t) {
                    return !t && !p(this._root) && this._fallbackRoot
                }, K.prototype._isSilentFallbackWarn = function(t) {
                    return this._silentFallbackWarn instanceof RegExp ? this._silentFallbackWarn.test(t) : this._silentFallbackWarn
                }, K.prototype._isSilentFallback = function(t, e) {
                    return this._isSilentFallbackWarn(e) && (this._isFallbackRoot() || t !== this.fallbackLocale)
                }, K.prototype._isSilentTranslationWarn = function(t) {
                    return this._silentTranslationWarn instanceof RegExp ? this._silentTranslationWarn.test(t) : this._silentTranslationWarn
                }, K.prototype._interpolate = function(t, e, n, a, i, o, r) {
                    if (!e) return null;
                    var l, s = this._path.getPathValue(e, n);
                    if (Array.isArray(s) || d(s)) return s;
                    if (p(s)) {
                        if (!d(e)) return null;
                        if (!c(l = e[n])) return null
                    } else {
                        if (!c(s)) return null;
                        l = s
                    }
                    return (l.indexOf("@:") >= 0 || l.indexOf("@.") >= 0) && (l = this._link(t, e, l, a, "raw", o, r)), this._render(l, i, o, n)
                }, K.prototype._link = function(t, e, n, a, i, o, r) {
                    var l = n,
                        s = l.match(Z);
                    for (var c in s)
                        if (s.hasOwnProperty(c)) {
                            var u = s[c],
                                d = u.match(V),
                                p = d[0],
                                f = d[1],
                                g = u.replace(p, "").replace(W, "");
                            if (m(r, g)) return l;
                            r.push(g);
                            var h = this._interpolate(t, e, g, a, "raw" === i ? "string" : i, "raw" === i ? void 0 : o, r);
                            if (this._isFallbackRoot(h)) {
                                if (!this._root) throw Error("unexpected error");
                                var C = this._root.$i18n;
                                h = C._translate(C._getMessages(), C.locale, C.fallbackLocale, g, a, i, o)
                            }
                            h = this._warnDefault(t, g, h, a, Array.isArray(o) ? o : [o], i), this._modifiers.hasOwnProperty(f) ? h = this._modifiers[f](h) : H.hasOwnProperty(f) && (h = H[f](h)), r.pop(), l = h ? l.replace(u, h) : l
                        }
                    return l
                }, K.prototype._render = function(t, e, n, a) {
                    var i = this._formatter.interpolate(t, n, a);
                    return i || (i = q.interpolate(t, n, a)), "string" !== e || c(i) ? i : i.join("")
                }, K.prototype._appendItemToChain = function(t, e, n) {
                    var a = !1;
                    return m(t, e) || (a = !0, e && (a = "!" !== e[e.length - 1], e = e.replace(/!/g, ""), t.push(e), n && n[e] && (a = n[e]))), a
                }, K.prototype._appendLocaleToChain = function(t, e, n) {
                    var a, i = e.split("-");
                    do {
                        var o = i.join("-");
                        a = this._appendItemToChain(t, o, n), i.splice(-1, 1)
                    } while (i.length && !0 === a);
                    return a
                }, K.prototype._appendBlockToChain = function(t, e, n) {
                    for (var a = !0, i = 0; i < e.length && "boolean" == typeof a; i++) {
                        var o = e[i];
                        c(o) && (a = this._appendLocaleToChain(t, o, n))
                    }
                    return a
                }, K.prototype._getLocaleChain = function(t, e) {
                    if ("" === t) return [];
                    this._localeChainCache || (this._localeChainCache = {});
                    var n = this._localeChainCache[t];
                    if (!n) {
                        e || (e = this.fallbackLocale), n = [];
                        for (var a, i = [t]; l(i);) i = this._appendBlockToChain(n, i, e);
                        (i = c(a = l(e) ? e : s(e) ? e.default ? e.default : null : e) ? [a] : a) && this._appendBlockToChain(n, i, null), this._localeChainCache[t] = n
                    }
                    return n
                }, K.prototype._translate = function(t, e, n, a, i, o, r) {
                    for (var l, s = this._getLocaleChain(e, n), c = 0; c < s.length; c++) {
                        var u = s[c];
                        if (!p(l = this._interpolate(u, t[u], a, i, o, r, [a]))) return l
                    }
                    return null
                }, K.prototype._t = function(t, e, n, a) {
                    for (var i, o = [], r = arguments.length - 4; r-- > 0;) o[r] = arguments[r + 4];
                    if (!t) return "";
                    var l = f.apply(void 0, o),
                        s = l.locale || e,
                        c = this._translate(n, s, this.fallbackLocale, t, a, "string", l.params);
                    if (this._isFallbackRoot(c)) {
                        if (!this._root) throw Error("unexpected error");
                        return (i = this._root).$t.apply(i, [t].concat(o))
                    }
                    return c = this._warnDefault(s, t, c, a, o, "string"), this._postTranslation && null != c && (c = this._postTranslation(c, t)), c
                }, K.prototype.t = function(t) {
                    for (var e, n = [], a = arguments.length - 1; a-- > 0;) n[a] = arguments[a + 1];
                    return (e = this)._t.apply(e, [t, this.locale, this._getMessages(), null].concat(n))
                }, K.prototype._i = function(t, e, n, a, i) {
                    var o = this._translate(n, e, this.fallbackLocale, t, a, "raw", i);
                    if (this._isFallbackRoot(o)) {
                        if (!this._root) throw Error("unexpected error");
                        return this._root.$i18n.i(t, e, i)
                    }
                    return this._warnDefault(e, t, o, a, [i], "raw")
                }, K.prototype.i = function(t, e, n) {
                    return t ? (c(e) || (e = this.locale), this._i(t, e, this._getMessages(), null, n)) : ""
                }, K.prototype._tc = function(t, e, n, a, i) {
                    for (var o, r = [], l = arguments.length - 5; l-- > 0;) r[l] = arguments[l + 5];
                    if (!t) return "";
                    void 0 === i && (i = 1);
                    var s = {
                            count: i,
                            n: i
                        },
                        c = f.apply(void 0, r);
                    return c.params = Object.assign(s, c.params), r = null === c.locale ? [c.params] : [c.locale, c.params], this.fetchChoice((o = this)._t.apply(o, [t, e, n, a].concat(r)), i)
                }, K.prototype.fetchChoice = function(t, e) {
                    if (!t && !c(t)) return null;
                    var n = t.split("|");
                    return n[e = this.getChoiceIndex(e, n.length)] ? n[e].trim() : t
                }, K.prototype.tc = function(t, e) {
                    for (var n, a = [], i = arguments.length - 2; i-- > 0;) a[i] = arguments[i + 2];
                    return (n = this)._tc.apply(n, [t, this.locale, this._getMessages(), null, e].concat(a))
                }, K.prototype._te = function(t, e, n) {
                    for (var a = [], i = arguments.length - 3; i-- > 0;) a[i] = arguments[i + 3];
                    var o = f.apply(void 0, a).locale || e;
                    return this._exist(n[o], t)
                }, K.prototype.te = function(t, e) {
                    return this._te(t, this.locale, this._getMessages(), e)
                }, K.prototype.getLocaleMessage = function(t) {
                    return g(this._vm.messages[t] || {})
                }, K.prototype.setLocaleMessage = function(t, e) {
                    "warn" !== this._warnHtmlInMessage && "error" !== this._warnHtmlInMessage || this._checkLocaleMessage(t, this._warnHtmlInMessage, e), this._vm.$set(this._vm.messages, t, e)
                }, K.prototype.mergeLocaleMessage = function(t, e) {
                    "warn" !== this._warnHtmlInMessage && "error" !== this._warnHtmlInMessage || this._checkLocaleMessage(t, this._warnHtmlInMessage, e), this._vm.$set(this._vm.messages, t, y({}, this._vm.messages[t] || {}, e))
                }, K.prototype.getDateTimeFormat = function(t) {
                    return g(this._vm.dateTimeFormats[t] || {})
                }, K.prototype.setDateTimeFormat = function(t, e) {
                    this._vm.$set(this._vm.dateTimeFormats, t, e), this._clearDateTimeFormat(t, e)
                }, K.prototype.mergeDateTimeFormat = function(t, e) {
                    this._vm.$set(this._vm.dateTimeFormats, t, y(this._vm.dateTimeFormats[t] || {}, e)), this._clearDateTimeFormat(t, e)
                }, K.prototype._clearDateTimeFormat = function(t, e) {
                    for (var n in e) {
                        var a = t + "__" + n;
                        this._dateTimeFormatters.hasOwnProperty(a) && delete this._dateTimeFormatters[a]
                    }
                }, K.prototype._localizeDateTime = function(t, e, n, a, i) {
                    for (var o = e, r = a[o], l = this._getLocaleChain(e, n), s = 0; s < l.length; s++) {
                        var c = l[s];
                        if (o = c, !p(r = a[c]) && !p(r[i])) break
                    }
                    if (p(r) || p(r[i])) return null;
                    var u = r[i],
                        d = o + "__" + i,
                        f = this._dateTimeFormatters[d];
                    return f || (f = this._dateTimeFormatters[d] = new Intl.DateTimeFormat(o, u)), f.format(t)
                }, K.prototype._d = function(t, e, n) {
                    if (!n) return new Intl.DateTimeFormat(e).format(t);
                    var a = this._localizeDateTime(t, e, this.fallbackLocale, this._getDateTimeFormats(), n);
                    if (this._isFallbackRoot(a)) {
                        if (!this._root) throw Error("unexpected error");
                        return this._root.$i18n.d(t, n, e)
                    }
                    return a || ""
                }, K.prototype.d = function(t) {
                    for (var e = [], n = arguments.length - 1; n-- > 0;) e[n] = arguments[n + 1];
                    var a = this.locale,
                        i = null;
                    return 1 === e.length ? c(e[0]) ? i = e[0] : s(e[0]) && (e[0].locale && (a = e[0].locale), e[0].key && (i = e[0].key)) : 2 === e.length && (c(e[0]) && (i = e[0]), c(e[1]) && (a = e[1])), this._d(t, a, i)
                }, K.prototype.getNumberFormat = function(t) {
                    return g(this._vm.numberFormats[t] || {})
                }, K.prototype.setNumberFormat = function(t, e) {
                    this._vm.$set(this._vm.numberFormats, t, e), this._clearNumberFormat(t, e)
                }, K.prototype.mergeNumberFormat = function(t, e) {
                    this._vm.$set(this._vm.numberFormats, t, y(this._vm.numberFormats[t] || {}, e)), this._clearNumberFormat(t, e)
                }, K.prototype._clearNumberFormat = function(t, e) {
                    for (var n in e) {
                        var a = t + "__" + n;
                        this._numberFormatters.hasOwnProperty(a) && delete this._numberFormatters[a]
                    }
                }, K.prototype._getNumberFormatter = function(t, e, n, a, i, o) {
                    for (var r = e, l = a[r], s = this._getLocaleChain(e, n), c = 0; c < s.length; c++) {
                        var u = s[c];
                        if (r = u, !p(l = a[u]) && !p(l[i])) break
                    }
                    if (p(l) || p(l[i])) return null;
                    var d, f = l[i];
                    if (o) d = new Intl.NumberFormat(r, Object.assign({}, f, o));
                    else {
                        var g = r + "__" + i;
                        (d = this._numberFormatters[g]) || (d = this._numberFormatters[g] = new Intl.NumberFormat(r, f))
                    }
                    return d
                }, K.prototype._n = function(t, e, n, a) {
                    if (!K.availabilities.numberFormat) return "";
                    if (!n) return (a ? new Intl.NumberFormat(e, a) : new Intl.NumberFormat(e)).format(t);
                    var i = this._getNumberFormatter(t, e, this.fallbackLocale, this._getNumberFormats(), n, a),
                        o = i && i.format(t);
                    if (this._isFallbackRoot(o)) {
                        if (!this._root) throw Error("unexpected error");
                        return this._root.$i18n.n(t, Object.assign({}, {
                            key: n,
                            locale: e
                        }, a))
                    }
                    return o || ""
                }, K.prototype.n = function(t) {
                    for (var e = [], n = arguments.length - 1; n-- > 0;) e[n] = arguments[n + 1];
                    var a = this.locale,
                        i = null,
                        r = null;
                    return 1 === e.length ? c(e[0]) ? i = e[0] : s(e[0]) && (e[0].locale && (a = e[0].locale), e[0].key && (i = e[0].key), r = Object.keys(e[0]).reduce((function(t, n) {
                        var a;
                        return m(o, n) ? Object.assign({}, t, ((a = {})[n] = e[0][n], a)) : t
                    }), null)) : 2 === e.length && (c(e[0]) && (i = e[0]), c(e[1]) && (a = e[1])), this._n(t, a, i, r)
                }, K.prototype._ntp = function(t, e, n, a) {
                    if (!K.availabilities.numberFormat) return [];
                    if (!n) return (a ? new Intl.NumberFormat(e, a) : new Intl.NumberFormat(e)).formatToParts(t);
                    var i = this._getNumberFormatter(t, e, this.fallbackLocale, this._getNumberFormats(), n, a),
                        o = i && i.formatToParts(t);
                    if (this._isFallbackRoot(o)) {
                        if (!this._root) throw Error("unexpected error");
                        return this._root.$i18n._ntp(t, e, n, a)
                    }
                    return o || []
                }, Object.defineProperties(K.prototype, J), Object.defineProperty(K, "availabilities", {
                    get: function() {
                        if (!z) {
                            var t = "undefined" != typeof Intl;
                            z = {
                                dateTimeFormat: t && void 0 !== Intl.DateTimeFormat,
                                numberFormat: t && void 0 !== Intl.NumberFormat
                            }
                        }
                        return z
                    }
                }), K.install = E, K.version = "8.20.0";
                const X = K,
                    Y = {
                        en: {
                            editor: {
                                run: "Run",
                                errorInEditor: "Errors exist in code!",
                                chartOK: "Chart has been generated successfully, ",
                                darkMode: "Dark Mode",
                                enableDecal: "Decal Pattern",
                                renderCfgTitle: "Render",
                                renderer: "Renderer",
                                useDirtyRect: "Use Dirty Rect",
                                download: "Download",
                                edit: "Edit",
                                monacoMode: "Enable Type Checking",
                                tabEditor: "Edit Example",
                                tabFullCodePreview: "Full Code",
                                tabOptionPreview: "Option Preview",
                                minimalBundle: "Minimal Bundle"
                            },
                            chartTypes: {
                                line: "Line",
                                bar: "Bar",
                                pie: "Pie",
                                scatter: "Scatter",
                                map: "GEO/Map",
                                candlestick: "Candlestick",
                                radar: "Radar",
                                boxplot: "Boxplot",
                                heatmap: "Heatmap",
                                graph: "Graph",
                                lines: "Lines",
                                tree: "Tree",
                                treemap: "Treemap",
                                sunburst: "Sunburst",
                                parallel: "Parallel",
                                sankey: "Sankey",
                                funnel: "Funnel",
                                gauge: "Gauge",
                                pictorialBar: "PictorialBar",
                                themeRiver: "ThemeRiver",
                                calendar: "Calendar",
                                custom: "Custom",
                                dataset: "Dataset",
                                dataZoom: "DataZoom",
                                drag: "Drag",
                                rich: "Rich Text",
                                globe: "3D Globe",
                                bar3D: "3D Bar",
                                scatter3D: "3D Scatter",
                                surface: "3D Surface",
                                map3D: "3D Map",
                                lines3D: "3D Lines",
                                line3D: "3D Line",
                                scatterGL: "Scatter GL",
                                linesGL: "Lines GL",
                                flowGL: "Flow GL",
                                graphGL: "Graph GL"
                            }
                        },
                        zh: {
                            editor: {
                                run: "运行",
                                errorInEditor: "编辑器内容有误！",
                                chartOK: "图表已生成, ",
                                darkMode: "深色模式",
                                enableDecal: "无障碍花纹",
                                renderCfgTitle: "渲染设置",
                                useDirtyRect: "开启脏矩形优化",
                                renderer: "渲染模式",
                                download: "下载示例",
                                edit: "编辑",
                                monacoMode: "开启类型检查",
                                tabEditor: "示例编辑",
                                tabFullCodePreview: "完整代码",
                                tabOptionPreview: "配置项",
                                minimalBundle: "按需引入"
                            },
                            chartTypes: {
                                line: "折线图",
                                bar: "柱状图",
                                pie: "饼图",
                                scatter: "散点图",
                                map: "地理坐标/地图",
                                candlestick: "K 线图",
                                radar: "雷达图",
                                boxplot: "盒须图",
                                heatmap: "热力图",
                                graph: "关系图",
                                lines: "路径图",
                                tree: "树图",
                                treemap: "矩形树图",
                                sunburst: "旭日图",
                                parallel: "平行坐标系",
                                sankey: "桑基图",
                                funnel: "漏斗图",
                                gauge: "仪表盘",
                                pictorialBar: "象形柱图",
                                themeRiver: "主题河流图",
                                calendar: "日历坐标系",
                                custom: "自定义系列",
                                dataset: "数据集",
                                dataZoom: "数据区域缩放",
                                drag: "拖拽",
                                rich: "富文本",
                                globe: "3D 地球",
                                bar3D: "3D 柱状图",
                                scatter3D: "3D 散点图",
                                surface: "3D 曲面",
                                map3D: "3D 地图",
                                lines3D: "3D 路径图",
                                line3D: "3D 折线图",
                                scatterGL: "GL 散点图",
                                linesGL: "GL 路径图",
                                flowGL: "GL 矢量场图",
                                graphGL: "GL 关系图"
                            }
                        }
                    };
                var Q = function() {
                    var t = this,
                        e = t.$createElement,
                        n = t._self._c || e;
                    return n("div", {
                        attrs: {
                            id: "main-container"
                        }
                    }, [t.shared.isMobile ? t._e() : n("div", {
                        style: {
                            width: t.leftContainerSize + "%"
                        },
                        attrs: {
                            id: "editor-left-container"
                        }
                    }, [n("el-tabs", {
                        attrs: {
                            type: "border-card"
                        },
                        model: {
                            value: t.currentTab,
                            callback: function(e) {
                                t.currentTab = e
                            },
                            expression: "currentTab"
                        }
                    }, [n("el-tab-pane", {
                        attrs: {
                            label: t.$t("editor.tabEditor"),
                            name: "code-editor"
                        }
                    }, [n("el-container", [n("el-header", {
                        attrs: {
                            id: "editor-control-panel"
                        }
                    }, [n("div", {
                        attrs: {
                            id: "code-info"
                        }
                    }, [t.shared.editorStatus.message ? [n("span", {
                        staticClass: "code-info-time"
                    }, [t._v(t._s(t.currentTime))]), t._v(" "), n("span", {
                        class: "code-info-type-" + t.shared.editorStatus.type
                    }, [t._v(t._s(t.shared.editorStatus.message))])] : t._e()], 2), t._v(" "), n("div", {
                        staticClass: "editor-controls"
                    }, [n("a", {
                        staticClass: "btn btn-default btn-sm",
                        attrs: {
                            href: "javascript:;"
                        },
                        on: {
                            click: t.disposeAndRun
                        }
                    }, [t._v(t._s(t.$t("editor.run")))])])]), t._v(" "), n("el-main", [t.shared.typeCheck ? n("CodeMonaco", {
                        attrs: {
                            id: "code-panel",
                            initialCode: t.initialCode
                        }
                    }) : n("CodeAce", {
                        attrs: {
                            id: "code-panel",
                            initialCode: t.initialCode
                        }
                    })], 1)], 1)], 1), t._v(" "), n("el-tab-pane", {
                        attrs: {
                            label: t.$t("editor.tabFullCodePreview"),
                            name: "full-code",
                            lazy: !0
                        }
                    }, [n("el-container", {
                        staticStyle: {
                            width: "100%",
                            height: "100%"
                        }
                    }, [n("el-header", {
                        attrs: {
                            id: "full-code-generate-config"
                        }
                    }, [n("span", {
                        staticClass: "full-code-generate-config-label"
                    }), t._v(" "), n("el-switch", {
                        staticClass: "enable-decal",
                        attrs: {
                            "active-text": t.$t("editor.minimalBundle"),
                            "inactive-text": ""
                        },
                        model: {
                            value: t.fullCodeConfig.minimal,
                            callback: function(e) {
                                t.$set(t.fullCodeConfig, "minimal", e)
                            },
                            expression: "fullCodeConfig.minimal"
                        }
                    }), t._v(" "), n("el-switch", {
                        staticClass: "enable-decal",
                        attrs: {
                            "active-text": "ES Modules",
                            "inactive-text": ""
                        },
                        model: {
                            value: t.fullCodeConfig.esm,
                            callback: function(e) {
                                t.$set(t.fullCodeConfig, "esm", e)
                            },
                            expression: "fullCodeConfig.esm"
                        }
                    })], 1), t._v(" "), n("el-main", [n("FullCodePreview", {
                        attrs: {
                            code: t.fullCode
                        }
                    })], 1)], 1)], 1), t._v(" "), n("el-tab-pane", {
                        attrs: {
                            label: t.$t("editor.tabOptionPreview"),
                            name: "full-option"
                        }
                    }, [n("div", {
                        attrs: {
                            id: "option-outline"
                        }
                    })])], 1)], 1), t._v(" "), t.shared.isMobile ? t._e() : n("div", {
                        staticClass: "handler",
                        style: {
                            left: t.leftContainerSize + "%"
                        },
                        attrs: {
                            id: "h-handler"
                        },
                        on: {
                            mousedown: t.onSplitterDragStart
                        }
                    }), t._v(" "), n("Preview", {
                        ref: "preview",
                        staticClass: "right-container",
                        style: {
                            width: 100 - t.leftContainerSize + "%",
                            left: t.leftContainerSize + "%"
                        },
                        attrs: {
                            inEditor: !0
                        }
                    })], 1)
                };
                Q._withStripped = !0;
                var tt = function() {
                    var t = this,
                        e = t.$createElement;
                    return (t._self._c || e)("div", {
                        directives: [{
                            name: "loading",
                            rawName: "v-loading",
                            value: t.loading,
                            expression: "loading"
                        }],
                        staticClass: "ace-editor-main"
                    })
                };
                tt._withStripped = !0;
                var et = [{
                        name: "color",
                        count: 1835
                    }, {
                        name: "shadowColor",
                        count: 1770
                    }, {
                        name: "shadowBlur",
                        count: 1770
                    }, {
                        name: "shadowOffsetX",
                        count: 1770
                    }, {
                        name: "shadowOffsetY",
                        count: 1770
                    }, {
                        name: "borderColor",
                        count: 1451
                    }, {
                        name: "borderWidth",
                        count: 1450
                    }, {
                        name: "width",
                        count: 1411
                    }, {
                        name: "borderType",
                        count: 1385
                    }, {
                        name: "borderDashOffset",
                        count: 1373
                    }, {
                        name: "height",
                        count: 1120
                    }, {
                        name: "backgroundColor",
                        count: 1111
                    }, {
                        name: "fontSize",
                        count: 1098
                    }, {
                        name: "fontStyle",
                        count: 1081
                    }, {
                        name: "fontWeight",
                        count: 1081
                    }, {
                        name: "fontFamily",
                        count: 1081
                    }, {
                        name: "lineHeight",
                        count: 1081
                    }, {
                        name: "textBorderColor",
                        count: 1081
                    }, {
                        name: "textBorderWidth",
                        count: 1081
                    }, {
                        name: "textBorderType",
                        count: 1081
                    }, {
                        name: "textBorderDashOffset",
                        count: 1081
                    }, {
                        name: "textShadowColor",
                        count: 1081
                    }, {
                        name: "textShadowBlur",
                        count: 1081
                    }, {
                        name: "textShadowOffsetX",
                        count: 1081
                    }, {
                        name: "textShadowOffsetY",
                        count: 1081
                    }, {
                        name: "padding",
                        count: 1079
                    }, {
                        name: "borderRadius",
                        count: 1051
                    }, {
                        name: "align",
                        count: 916
                    }, {
                        name: "verticalAlign",
                        count: 913
                    }, {
                        name: "opacity",
                        count: 692
                    }, {
                        name: "show",
                        count: 664
                    }, {
                        name: "overflow",
                        count: 567
                    }, {
                        name: "ellipsis",
                        count: 567
                    }, {
                        name: "lineOverflow",
                        count: 567
                    }, {
                        name: "position",
                        count: 528
                    }, {
                        name: "rich",
                        count: 514
                    }, {
                        name: "<style_name>",
                        count: 514
                    }, {
                        name: "distance",
                        count: 472
                    }, {
                        name: "label",
                        count: 468
                    }, {
                        name: "type",
                        count: 389
                    }, {
                        name: "rotate",
                        count: 365
                    }, {
                        name: "offset",
                        count: 357
                    }, {
                        name: "itemStyle",
                        count: 356
                    }, {
                        name: "borderCap",
                        count: 347
                    }, {
                        name: "borderJoin",
                        count: 347
                    }, {
                        name: "borderMiterLimit",
                        count: 347
                    }, {
                        name: "formatter",
                        count: 331
                    }, {
                        name: "lineStyle",
                        count: 298
                    }, {
                        name: "dashOffset",
                        count: 278
                    }, {
                        name: "cap",
                        count: 278
                    }, {
                        name: "join",
                        count: 278
                    }, {
                        name: "miterLimit",
                        count: 278
                    }, {
                        name: "emphasis",
                        count: 175
                    }, {
                        name: "blur",
                        count: 143
                    }, {
                        name: "name",
                        count: 133
                    }, {
                        name: "curveness",
                        count: 124
                    }, {
                        name: "symbol",
                        count: 119
                    }, {
                        name: "symbolSize",
                        count: 119
                    }, {
                        name: "x",
                        count: 115
                    }, {
                        name: "y",
                        count: 115
                    }, {
                        name: "value",
                        count: 101
                    }, {
                        name: "symbolKeepAspect",
                        count: 94
                    }, {
                        name: "silent",
                        count: 93
                    }, {
                        name: "labelLine",
                        count: 81
                    }, {
                        name: "rotation",
                        count: 78
                    }, {
                        name: "symbolOffset",
                        count: 75
                    }, {
                        name: "id",
                        count: 71
                    }, {
                        name: "data",
                        count: 71
                    }, {
                        name: "symbolRotate",
                        count: 67
                    }, {
                        name: "animationDuration",
                        count: 66
                    }, {
                        name: "animationEasing",
                        count: 66
                    }, {
                        name: "animationDelay",
                        count: 65
                    }, {
                        name: "z",
                        count: 64
                    }, {
                        name: "animation",
                        count: 64
                    }, {
                        name: "animationDurationUpdate",
                        count: 63
                    }, {
                        name: "animationThreshold",
                        count: 62
                    }, {
                        name: "animationEasingUpdate",
                        count: 62
                    }, {
                        name: "animationDelayUpdate",
                        count: 62
                    }, {
                        name: "style",
                        count: 60
                    }, {
                        name: "select",
                        count: 56
                    }, {
                        name: "textStyle",
                        count: 54
                    }, {
                        name: "zlevel",
                        count: 52
                    }, {
                        name: "transition",
                        count: 48
                    }, {
                        name: "focus",
                        count: 41
                    }, {
                        name: "blurScope",
                        count: 41
                    }, {
                        name: "coord",
                        count: 41
                    }, {
                        name: "tooltip",
                        count: 40
                    }, {
                        name: "inside",
                        count: 40
                    }, {
                        name: "valueIndex",
                        count: 40
                    }, {
                        name: "valueDim",
                        count: 40
                    }, {
                        name: "extraCssText",
                        count: 38
                    }, {
                        name: "interval",
                        count: 34
                    }, {
                        name: "left",
                        count: 33
                    }, {
                        name: "top",
                        count: 33
                    }, {
                        name: "right",
                        count: 33
                    }, {
                        name: "bottom",
                        count: 33
                    }, {
                        name: "draggable",
                        count: 31
                    }, {
                        name: "decal",
                        count: 28
                    }, {
                        name: "dashArrayX",
                        count: 28
                    }, {
                        name: "dashArrayY",
                        count: 28
                    }, {
                        name: "maxTileWidth",
                        count: 28
                    }, {
                        name: "maxTileHeight",
                        count: 28
                    }, {
                        name: "margin",
                        count: 27
                    }, {
                        name: "xAxis",
                        count: 27
                    }, {
                        name: "yAxis",
                        count: 27
                    }, {
                        name: "origin",
                        count: 26
                    }, {
                        name: "0",
                        count: 26
                    }, {
                        name: "1",
                        count: 26
                    }, {
                        name: "precision",
                        count: 25
                    }, {
                        name: "scaleX",
                        count: 25
                    }, {
                        name: "scaleY",
                        count: 25
                    }, {
                        name: "originX",
                        count: 25
                    }, {
                        name: "originY",
                        count: 25
                    }, {
                        name: "info",
                        count: 25
                    }, {
                        name: "invisible",
                        count: 25
                    }, {
                        name: "ignore",
                        count: 25
                    }, {
                        name: "textContent",
                        count: 25
                    }, {
                        name: "textConfig",
                        count: 25
                    }, {
                        name: "layoutRect",
                        count: 25
                    }, {
                        name: "local",
                        count: 25
                    }, {
                        name: "insideFill",
                        count: 25
                    }, {
                        name: "insideStroke",
                        count: 25
                    }, {
                        name: "outsideFill",
                        count: 25
                    }, {
                        name: "outsideStroke",
                        count: 25
                    }, {
                        name: "smooth",
                        count: 24
                    }, {
                        name: "selectedMode",
                        count: 23
                    }, {
                        name: "fill",
                        count: 23
                    }, {
                        name: "stroke",
                        count: 23
                    }, {
                        name: "lineWidth",
                        count: 23
                    }, {
                        name: "length",
                        count: 21
                    }, {
                        name: "areaStyle",
                        count: 20
                    }, {
                        name: "shape",
                        count: 20
                    }, {
                        name: "cursor",
                        count: 19
                    }, {
                        name: "showAbove",
                        count: 19
                    }, {
                        name: "splitNumber",
                        count: 18
                    }, {
                        name: "progressive",
                        count: 18
                    }, {
                        name: "length2",
                        count: 18
                    }, {
                        name: "minTurnAngle",
                        count: 18
                    }, {
                        name: "labelLayout",
                        count: 18
                    }, {
                        name: "hideOverlap",
                        count: 17
                    }, {
                        name: "moveOverlap",
                        count: 17
                    }, {
                        name: "dx",
                        count: 17
                    }, {
                        name: "dy",
                        count: 17
                    }, {
                        name: "labelLinePoints",
                        count: 17
                    }, {
                        name: "icon",
                        count: 16
                    }, {
                        name: "xAxisIndex",
                        count: 15
                    }, {
                        name: "yAxisIndex",
                        count: 15
                    }, {
                        name: "min",
                        count: 14
                    }, {
                        name: "max",
                        count: 14
                    }, {
                        name: "scale",
                        count: 14
                    }, {
                        name: "coordinateSystem",
                        count: 13
                    }, {
                        name: "markPoint",
                        count: 13
                    }, {
                        name: "markLine",
                        count: 13
                    }, {
                        name: "markArea",
                        count: 13
                    }, {
                        name: "z2",
                        count: 13
                    }, {
                        name: "during",
                        count: 13
                    }, {
                        name: "extra",
                        count: 13
                    }, {
                        name: "orient",
                        count: 12
                    }, {
                        name: "iconStyle",
                        count: 12
                    }, {
                        name: "areaColor",
                        count: 12
                    }, {
                        name: "$action",
                        count: 12
                    }, {
                        name: "bounding",
                        count: 12
                    }, {
                        name: "onclick",
                        count: 12
                    }, {
                        name: "onmouseover",
                        count: 12
                    }, {
                        name: "onmouseout",
                        count: 12
                    }, {
                        name: "onmousemove",
                        count: 12
                    }, {
                        name: "onmousewheel",
                        count: 12
                    }, {
                        name: "onmousedown",
                        count: 12
                    }, {
                        name: "onmouseup",
                        count: 12
                    }, {
                        name: "ondrag",
                        count: 12
                    }, {
                        name: "ondragstart",
                        count: 12
                    }, {
                        name: "ondragend",
                        count: 12
                    }, {
                        name: "ondragenter",
                        count: 12
                    }, {
                        name: "ondragleave",
                        count: 12
                    }, {
                        name: "ondragover",
                        count: 12
                    }, {
                        name: "ondrop",
                        count: 12
                    }, {
                        name: "legendHoverLink",
                        count: 12
                    }, {
                        name: "upperLabel",
                        count: 12
                    }, {
                        name: "dimensions",
                        count: 11
                    }, {
                        name: "axisPointer",
                        count: 10
                    }, {
                        name: "snap",
                        count: 10
                    }, {
                        name: "shadowStyle",
                        count: 10
                    }, {
                        name: "r",
                        count: 10
                    }, {
                        name: "encode",
                        count: 10
                    }, {
                        name: "minAngle",
                        count: 10
                    }, {
                        name: "morph",
                        count: 10
                    }, {
                        name: "title",
                        count: 9
                    }, {
                        name: "textAlign",
                        count: 9
                    }, {
                        name: "triggerEvent",
                        count: 9
                    }, {
                        name: "inverse",
                        count: 9
                    }, {
                        name: "axisLine",
                        count: 9
                    }, {
                        name: "axisTick",
                        count: 9
                    }, {
                        name: "axisLabel",
                        count: 9
                    }, {
                        name: "boundaryGap",
                        count: 8
                    }, {
                        name: "showMinLabel",
                        count: 8
                    }, {
                        name: "showMaxLabel",
                        count: 8
                    }, {
                        name: "splitLine",
                        count: 8
                    }, {
                        name: "size",
                        count: 8
                    }, {
                        name: "throttle",
                        count: 8
                    }, {
                        name: "center",
                        count: 8
                    }, {
                        name: "startAngle",
                        count: 8
                    }, {
                        name: "geoIndex",
                        count: 8
                    }, {
                        name: "cx",
                        count: 8
                    }, {
                        name: "cy",
                        count: 8
                    }, {
                        name: "seriesLayoutBy",
                        count: 8
                    }, {
                        name: "datasetIndex",
                        count: 8
                    }, {
                        name: "color0",
                        count: 8
                    }, {
                        name: "borderColor0",
                        count: 8
                    }, {
                        name: "nameGap",
                        count: 7
                    }, {
                        name: "minInterval",
                        count: 7
                    }, {
                        name: "maxInterval",
                        count: 7
                    }, {
                        name: "logBase",
                        count: 7
                    }, {
                        name: "alignWithLabel",
                        count: 7
                    }, {
                        name: "minorTick",
                        count: 7
                    }, {
                        name: "polarIndex",
                        count: 7
                    }, {
                        name: "clockwise",
                        count: 7
                    }, {
                        name: "clip",
                        count: 7
                    }, {
                        name: "text",
                        count: 6
                    }, {
                        name: "nameLocation",
                        count: 6
                    }, {
                        name: "nameTextStyle",
                        count: 6
                    }, {
                        name: "nameRotate",
                        count: 6
                    }, {
                        name: "splitArea",
                        count: 6
                    }, {
                        name: "triggerTooltip",
                        count: 6
                    }, {
                        name: "status",
                        count: 6
                    }, {
                        name: "handle",
                        count: 6
                    }, {
                        name: "textPosition",
                        count: 6
                    }, {
                        name: "textFill",
                        count: 6
                    }, {
                        name: "textBackgroundColor",
                        count: 6
                    }, {
                        name: "textBorderRadius",
                        count: 6
                    }, {
                        name: "textPadding",
                        count: 6
                    }, {
                        name: "line",
                        count: 6
                    }, {
                        name: "layout",
                        count: 6
                    }, {
                        name: "r0",
                        count: 6
                    }, {
                        name: "progressiveThreshold",
                        count: 6
                    }, {
                        name: "colorAlpha",
                        count: 6
                    }, {
                        name: "colorSaturation",
                        count: 6
                    }, {
                        name: "offsetCenter",
                        count: 6
                    }, {
                        name: "target",
                        count: 5
                    }, {
                        name: "itemGap",
                        count: 5
                    }, {
                        name: "minorSplitLine",
                        count: 5
                    }, {
                        name: "radius",
                        count: 5
                    }, {
                        name: "realtime",
                        count: 5
                    }, {
                        name: "zoom",
                        count: 5
                    }, {
                        name: "bar",
                        count: 5
                    }, {
                        name: "stack",
                        count: 5
                    }, {
                        name: "roam",
                        count: 5
                    }, {
                        name: "endAngle",
                        count: 5
                    }, {
                        name: "valueAnimation",
                        count: 5
                    }, {
                        name: "calendarIndex",
                        count: 5
                    }, {
                        name: "link",
                        count: 4
                    }, {
                        name: "selected",
                        count: 4
                    }, {
                        name: "trigger",
                        count: 4
                    }, {
                        name: "axis",
                        count: 4
                    }, {
                        name: "crossStyle",
                        count: 4
                    }, {
                        name: "end",
                        count: 4
                    }, {
                        name: "seriesIndex",
                        count: 4
                    }, {
                        name: "inRange",
                        count: 4
                    }, {
                        name: "outOfRange",
                        count: 4
                    }, {
                        name: "nameMap",
                        count: 4
                    }, {
                        name: "points",
                        count: 4
                    }, {
                        name: "smoothConstraint",
                        count: 4
                    }, {
                        name: "x1",
                        count: 4
                    }, {
                        name: "y1",
                        count: 4
                    }, {
                        name: "x2",
                        count: 4
                    }, {
                        name: "y2",
                        count: 4
                    }, {
                        name: "percent",
                        count: 4
                    }, {
                        name: "endLabel",
                        count: 4
                    }, {
                        name: "large",
                        count: 4
                    }, {
                        name: "largeThreshold",
                        count: 4
                    }, {
                        name: "hoverAnimation",
                        count: 4
                    }, {
                        name: "edgeLabel",
                        count: 4
                    }, {
                        name: "textVerticalAlign",
                        count: 3
                    }, {
                        name: "itemWidth",
                        count: 3
                    }, {
                        name: "itemHeight",
                        count: 3
                    }, {
                        name: "filterMode",
                        count: 3
                    }, {
                        name: "handleStyle",
                        count: 3
                    }, {
                        name: "brushStyle",
                        count: 3
                    }, {
                        name: "rect",
                        count: 3
                    }, {
                        name: "polygon",
                        count: 3
                    }, {
                        name: "map",
                        count: 3
                    }, {
                        name: "children",
                        count: 3
                    }, {
                        name: "image",
                        count: 3
                    }, {
                        name: "font",
                        count: 3
                    }, {
                        name: "source",
                        count: 3
                    }, {
                        name: "config",
                        count: 3
                    }, {
                        name: "print",
                        count: 3
                    }, {
                        name: "sort",
                        count: 3
                    }, {
                        name: "withName",
                        count: 3
                    }, {
                        name: "withoutName",
                        count: 3
                    }, {
                        name: "roundCap",
                        count: 3
                    }, {
                        name: "barWidth",
                        count: 3
                    }, {
                        name: "barMaxWidth",
                        count: 3
                    }, {
                        name: "barMinWidth",
                        count: 3
                    }, {
                        name: "progressiveChunkMode",
                        count: 3
                    }, {
                        name: "visualDimension",
                        count: 3
                    }, {
                        name: "visualMin",
                        count: 3
                    }, {
                        name: "visualMax",
                        count: 3
                    }, {
                        name: "colorMappingBy",
                        count: 3
                    }, {
                        name: "visibleMin",
                        count: 3
                    }, {
                        name: "childrenVisibleMin",
                        count: 3
                    }, {
                        name: "gapWidth",
                        count: 3
                    }, {
                        name: "borderColorSaturation",
                        count: 3
                    }, {
                        name: "levels",
                        count: 3
                    }, {
                        name: "selectorLabel",
                        count: 2
                    }, {
                        name: "gridIndex",
                        count: 2
                    }, {
                        name: "realtimeSort",
                        count: 2
                    }, {
                        name: "sortSeriesIndex",
                        count: 2
                    }, {
                        name: "onZero",
                        count: 2
                    }, {
                        name: "onZeroAxisIndex",
                        count: 2
                    }, {
                        name: "radar",
                        count: 2
                    }, {
                        name: "dataZoom",
                        count: 2
                    }, {
                        name: "radiusAxisIndex",
                        count: 2
                    }, {
                        name: "angleAxisIndex",
                        count: 2
                    }, {
                        name: "start",
                        count: 2
                    }, {
                        name: "startValue",
                        count: 2
                    }, {
                        name: "endValue",
                        count: 2
                    }, {
                        name: "minSpan",
                        count: 2
                    }, {
                        name: "maxSpan",
                        count: 2
                    }, {
                        name: "minValueSpan",
                        count: 2
                    }, {
                        name: "maxValueSpan",
                        count: 2
                    }, {
                        name: "zoomLock",
                        count: 2
                    }, {
                        name: "rangeMode",
                        count: 2
                    }, {
                        name: "handleIcon",
                        count: 2
                    }, {
                        name: "handleSize",
                        count: 2
                    }, {
                        name: "moveHandleStyle",
                        count: 2
                    }, {
                        name: "range",
                        count: 2
                    }, {
                        name: "textGap",
                        count: 2
                    }, {
                        name: "dimension",
                        count: 2
                    }, {
                        name: "hoverLink",
                        count: 2
                    }, {
                        name: "controller",
                        count: 2
                    }, {
                        name: "categories",
                        count: 2
                    }, {
                        name: "triggerOn",
                        count: 2
                    }, {
                        name: "toolbox",
                        count: 2
                    }, {
                        name: "itemSize",
                        count: 2
                    }, {
                        name: "back",
                        count: 2
                    }, {
                        name: "option",
                        count: 2
                    }, {
                        name: "brush",
                        count: 2
                    }, {
                        name: "lineX",
                        count: 2
                    }, {
                        name: "lineY",
                        count: 2
                    }, {
                        name: "keep",
                        count: 2
                    }, {
                        name: "clear",
                        count: 2
                    }, {
                        name: "brushType",
                        count: 2
                    }, {
                        name: "aspectScale",
                        count: 2
                    }, {
                        name: "boundingCoords",
                        count: 2
                    }, {
                        name: "scaleLimit",
                        count: 2
                    }, {
                        name: "nameProperty",
                        count: 2
                    }, {
                        name: "layoutCenter",
                        count: 2
                    }, {
                        name: "layoutSize",
                        count: 2
                    }, {
                        name: "parallel",
                        count: 2
                    }, {
                        name: "parallelIndex",
                        count: 2
                    }, {
                        name: "loop",
                        count: 2
                    }, {
                        name: "checkpointStyle",
                        count: 2
                    }, {
                        name: "controlStyle",
                        count: 2
                    }, {
                        name: "progress",
                        count: 2
                    }, {
                        name: "diffChildrenByName",
                        count: 2
                    }, {
                        name: "polyline",
                        count: 2
                    }, {
                        name: "cpx1",
                        count: 2
                    }, {
                        name: "cpy1",
                        count: 2
                    }, {
                        name: "cpx2",
                        count: 2
                    }, {
                        name: "cpy2",
                        count: 2
                    }, {
                        name: "enabled",
                        count: 2
                    }, {
                        name: "series",
                        count: 2
                    }, {
                        name: "maxCount",
                        count: 2
                    }, {
                        name: "prefix",
                        count: 2
                    }, {
                        name: "separator",
                        count: 2
                    }, {
                        name: "middle",
                        count: 2
                    }, {
                        name: "sampling",
                        count: 2
                    }, {
                        name: "barMinHeight",
                        count: 2
                    }, {
                        name: "barMinAngle",
                        count: 2
                    }, {
                        name: "barGap",
                        count: 2
                    }, {
                        name: "barCategoryGap",
                        count: 2
                    }, {
                        name: "period",
                        count: 2
                    }, {
                        name: "nodeClick",
                        count: 2
                    }, {
                        name: "nodes",
                        count: 2
                    }, {
                        name: "links",
                        count: 2
                    }, {
                        name: "edges",
                        count: 2
                    }, {
                        name: "depth",
                        count: 2
                    }, {
                        name: "detail",
                        count: 2
                    }, {
                        name: "keepAspect",
                        count: 2
                    }, {
                        name: "symbolPosition",
                        count: 2
                    }, {
                        name: "symbolRepeat",
                        count: 2
                    }, {
                        name: "symbolRepeatDirection",
                        count: 2
                    }, {
                        name: "symbolMargin",
                        count: 2
                    }, {
                        name: "symbolClip",
                        count: 2
                    }, {
                        name: "symbolBoundingData",
                        count: 2
                    }, {
                        name: "symbolPatternSize",
                        count: 2
                    }, {
                        name: "subtext",
                        count: 1
                    }, {
                        name: "sublink",
                        count: 1
                    }, {
                        name: "subtarget",
                        count: 1
                    }, {
                        name: "subtextStyle",
                        count: 1
                    }, {
                        name: "legend",
                        count: 1
                    }, {
                        name: "inactiveColor",
                        count: 1
                    }, {
                        name: "scrollDataIndex",
                        count: 1
                    }, {
                        name: "pageButtonItemGap",
                        count: 1
                    }, {
                        name: "pageButtonGap",
                        count: 1
                    }, {
                        name: "pageButtonPosition",
                        count: 1
                    }, {
                        name: "pageFormatter",
                        count: 1
                    }, {
                        name: "pageIcons",
                        count: 1
                    }, {
                        name: "horizontal",
                        count: 1
                    }, {
                        name: "vertical",
                        count: 1
                    }, {
                        name: "pageIconColor",
                        count: 1
                    }, {
                        name: "pageIconInactiveColor",
                        count: 1
                    }, {
                        name: "pageIconSize",
                        count: 1
                    }, {
                        name: "pageTextStyle",
                        count: 1
                    }, {
                        name: "selector",
                        count: 1
                    }, {
                        name: "selectorPosition",
                        count: 1
                    }, {
                        name: "selectorItemGap",
                        count: 1
                    }, {
                        name: "selectorButtonGap",
                        count: 1
                    }, {
                        name: "grid",
                        count: 1
                    }, {
                        name: "containLabel",
                        count: 1
                    }, {
                        name: "polar",
                        count: 1
                    }, {
                        name: "radiusAxis",
                        count: 1
                    }, {
                        name: "angleAxis",
                        count: 1
                    }, {
                        name: "indicator",
                        count: 1
                    }, {
                        name: "disabled",
                        count: 1
                    }, {
                        name: "zoomOnMouseWheel",
                        count: 1
                    }, {
                        name: "moveOnMouseMove",
                        count: 1
                    }, {
                        name: "moveOnMouseWheel",
                        count: 1
                    }, {
                        name: "preventDefaultMouseMove",
                        count: 1
                    }, {
                        name: "slider",
                        count: 1
                    }, {
                        name: "dataBackground",
                        count: 1
                    }, {
                        name: "selectedDataBackground",
                        count: 1
                    }, {
                        name: "fillerColor",
                        count: 1
                    }, {
                        name: "moveHandleIcon",
                        count: 1
                    }, {
                        name: "moveHandleSize",
                        count: 1
                    }, {
                        name: "labelPrecision",
                        count: 1
                    }, {
                        name: "labelFormatter",
                        count: 1
                    }, {
                        name: "showDetail",
                        count: 1
                    }, {
                        name: "showDataShadow",
                        count: 1
                    }, {
                        name: "brushSelect",
                        count: 1
                    }, {
                        name: "visualMap",
                        count: 1
                    }, {
                        name: "continuous",
                        count: 1
                    }, {
                        name: "calculable",
                        count: 1
                    }, {
                        name: "indicatorIcon",
                        count: 1
                    }, {
                        name: "indicatorSize",
                        count: 1
                    }, {
                        name: "indicatorStyle",
                        count: 1
                    }, {
                        name: "piecewise",
                        count: 1
                    }, {
                        name: "pieces",
                        count: 1
                    }, {
                        name: "minOpen",
                        count: 1
                    }, {
                        name: "maxOpen",
                        count: 1
                    }, {
                        name: "showLabel",
                        count: 1
                    }, {
                        name: "itemSymbol",
                        count: 1
                    }, {
                        name: "showContent",
                        count: 1
                    }, {
                        name: "alwaysShowContent",
                        count: 1
                    }, {
                        name: "showDelay",
                        count: 1
                    }, {
                        name: "hideDelay",
                        count: 1
                    }, {
                        name: "enterable",
                        count: 1
                    }, {
                        name: "renderMode",
                        count: 1
                    }, {
                        name: "confine",
                        count: 1
                    }, {
                        name: "appendToBody",
                        count: 1
                    }, {
                        name: "className",
                        count: 1
                    }, {
                        name: "transitionDuration",
                        count: 1
                    }, {
                        name: "order",
                        count: 1
                    }, {
                        name: "showTitle",
                        count: 1
                    }, {
                        name: "feature",
                        count: 1
                    }, {
                        name: "saveAsImage",
                        count: 1
                    }, {
                        name: "connectedBackgroundColor",
                        count: 1
                    }, {
                        name: "excludeComponents",
                        count: 1
                    }, {
                        name: "pixelRatio",
                        count: 1
                    }, {
                        name: "restore",
                        count: 1
                    }, {
                        name: "dataView",
                        count: 1
                    }, {
                        name: "readOnly",
                        count: 1
                    }, {
                        name: "optionToContent",
                        count: 1
                    }, {
                        name: "contentToOption",
                        count: 1
                    }, {
                        name: "lang",
                        count: 1
                    }, {
                        name: "textareaColor",
                        count: 1
                    }, {
                        name: "textareaBorderColor",
                        count: 1
                    }, {
                        name: "textColor",
                        count: 1
                    }, {
                        name: "buttonColor",
                        count: 1
                    }, {
                        name: "buttonTextColor",
                        count: 1
                    }, {
                        name: "magicType",
                        count: 1
                    }, {
                        name: "tiled",
                        count: 1
                    }, {
                        name: "brushLink",
                        count: 1
                    }, {
                        name: "brushMode",
                        count: 1
                    }, {
                        name: "transformable",
                        count: 1
                    }, {
                        name: "throttleType",
                        count: 1
                    }, {
                        name: "throttleDelay",
                        count: 1
                    }, {
                        name: "removeOnClick",
                        count: 1
                    }, {
                        name: "inBrush",
                        count: 1
                    }, {
                        name: "outOfBrush",
                        count: 1
                    }, {
                        name: "geo",
                        count: 1
                    }, {
                        name: "regions",
                        count: 1
                    }, {
                        name: "axisExpandable",
                        count: 1
                    }, {
                        name: "axisExpandCenter",
                        count: 1
                    }, {
                        name: "axisExpandCount",
                        count: 1
                    }, {
                        name: "axisExpandWidth",
                        count: 1
                    }, {
                        name: "axisExpandTriggerOn",
                        count: 1
                    }, {
                        name: "parallelAxisDefault",
                        count: 1
                    }, {
                        name: "parallelAxis",
                        count: 1
                    }, {
                        name: "dim",
                        count: 1
                    }, {
                        name: "areaSelectStyle",
                        count: 1
                    }, {
                        name: "singleAxis",
                        count: 1
                    }, {
                        name: "timeline",
                        count: 1
                    }, {
                        name: "axisType",
                        count: 1
                    }, {
                        name: "currentIndex",
                        count: 1
                    }, {
                        name: "autoPlay",
                        count: 1
                    }, {
                        name: "rewind",
                        count: 1
                    }, {
                        name: "playInterval",
                        count: 1
                    }, {
                        name: "replaceMerge",
                        count: 1
                    }, {
                        name: "controlPosition",
                        count: 1
                    }, {
                        name: "showPlayBtn",
                        count: 1
                    }, {
                        name: "showPrevBtn",
                        count: 1
                    }, {
                        name: "showNextBtn",
                        count: 1
                    }, {
                        name: "playIcon",
                        count: 1
                    }, {
                        name: "stopIcon",
                        count: 1
                    }, {
                        name: "prevIcon",
                        count: 1
                    }, {
                        name: "nextIcon",
                        count: 1
                    }, {
                        name: "graphic",
                        count: 1
                    }, {
                        name: "elements",
                        count: 1
                    }, {
                        name: "group",
                        count: 1
                    }, {
                        name: "circle",
                        count: 1
                    }, {
                        name: "ring",
                        count: 1
                    }, {
                        name: "sector",
                        count: 1
                    }, {
                        name: "arc",
                        count: 1
                    }, {
                        name: "bezierCurve",
                        count: 1
                    }, {
                        name: "calendar",
                        count: 1
                    }, {
                        name: "cellSize",
                        count: 1
                    }, {
                        name: "dayLabel",
                        count: 1
                    }, {
                        name: "firstDay",
                        count: 1
                    }, {
                        name: "monthLabel",
                        count: 1
                    }, {
                        name: "yearLabel",
                        count: 1
                    }, {
                        name: "dataset",
                        count: 1
                    }, {
                        name: "sourceHeader",
                        count: 1
                    }, {
                        name: "transform",
                        count: 1
                    }, {
                        name: "filter",
                        count: 1
                    }, {
                        name: "xxx:xxx",
                        count: 1
                    }, {
                        name: "fromDatasetIndex",
                        count: 1
                    }, {
                        name: "fromDatasetId",
                        count: 1
                    }, {
                        name: "fromTransformResult",
                        count: 1
                    }, {
                        name: "aria",
                        count: 1
                    }, {
                        name: "description",
                        count: 1
                    }, {
                        name: "general",
                        count: 1
                    }, {
                        name: "withTitle",
                        count: 1
                    }, {
                        name: "withoutTitle",
                        count: 1
                    }, {
                        name: "single",
                        count: 1
                    }, {
                        name: "multiple",
                        count: 1
                    }, {
                        name: "allData",
                        count: 1
                    }, {
                        name: "partialData",
                        count: 1
                    }, {
                        name: "decals",
                        count: 1
                    }, {
                        name: "showSymbol",
                        count: 1
                    }, {
                        name: "showAllSymbol",
                        count: 1
                    }, {
                        name: "connectNulls",
                        count: 1
                    }, {
                        name: "step",
                        count: 1
                    }, {
                        name: "smoothMonotone",
                        count: 1
                    }, {
                        name: "showBackground",
                        count: 1
                    }, {
                        name: "backgroundStyle",
                        count: 1
                    }, {
                        name: "pie",
                        count: 1
                    }, {
                        name: "selectedOffset",
                        count: 1
                    }, {
                        name: "minShowLabelAngle",
                        count: 1
                    }, {
                        name: "roseType",
                        count: 1
                    }, {
                        name: "avoidLabelOverlap",
                        count: 1
                    }, {
                        name: "stillShowZeroSum",
                        count: 1
                    }, {
                        name: "alignTo",
                        count: 1
                    }, {
                        name: "edgeDistance",
                        count: 1
                    }, {
                        name: "bleedMargin",
                        count: 1
                    }, {
                        name: "distanceToLabelLine",
                        count: 1
                    }, {
                        name: "maxSurfaceAngle",
                        count: 1
                    }, {
                        name: "scaleSize",
                        count: 1
                    }, {
                        name: "animationType",
                        count: 1
                    }, {
                        name: "animationTypeUpdate",
                        count: 1
                    }, {
                        name: "scatter",
                        count: 1
                    }, {
                        name: "effectScatter",
                        count: 1
                    }, {
                        name: "effectType",
                        count: 1
                    }, {
                        name: "showEffectOn",
                        count: 1
                    }, {
                        name: "rippleEffect",
                        count: 1
                    }, {
                        name: "radarIndex",
                        count: 1
                    }, {
                        name: "tree",
                        count: 1
                    }, {
                        name: "edgeShape",
                        count: 1
                    }, {
                        name: "edgeForkPosition",
                        count: 1
                    }, {
                        name: "expandAndCollapse",
                        count: 1
                    }, {
                        name: "initialTreeDepth",
                        count: 1
                    }, {
                        name: "leaves",
                        count: 1
                    }, {
                        name: "treemap",
                        count: 1
                    }, {
                        name: "squareRatio",
                        count: 1
                    }, {
                        name: "leafDepth",
                        count: 1
                    }, {
                        name: "drillDownIcon",
                        count: 1
                    }, {
                        name: "zoomToNodeRatio",
                        count: 1
                    }, {
                        name: "breadcrumb",
                        count: 1
                    }, {
                        name: "emptyItemWidth",
                        count: 1
                    }, {
                        name: "sunburst",
                        count: 1
                    }, {
                        name: "renderLabelForZeroData",
                        count: 1
                    }, {
                        name: "boxplot",
                        count: 1
                    }, {
                        name: "boxWidth",
                        count: 1
                    }, {
                        name: "candlestick",
                        count: 1
                    }, {
                        name: "heatmap",
                        count: 1
                    }, {
                        name: "pointSize",
                        count: 1
                    }, {
                        name: "blurSize",
                        count: 1
                    }, {
                        name: "minOpacity",
                        count: 1
                    }, {
                        name: "maxOpacity",
                        count: 1
                    }, {
                        name: "mapValueCalculation",
                        count: 1
                    }, {
                        name: "showLegendSymbol",
                        count: 1
                    }, {
                        name: "inactiveOpacity",
                        count: 1
                    }, {
                        name: "activeOpacity",
                        count: 1
                    }, {
                        name: "lines",
                        count: 1
                    }, {
                        name: "effect",
                        count: 1
                    }, {
                        name: "delay",
                        count: 1
                    }, {
                        name: "constantSpeed",
                        count: 1
                    }, {
                        name: "trailLength",
                        count: 1
                    }, {
                        name: "coords",
                        count: 1
                    }, {
                        name: "graph",
                        count: 1
                    }, {
                        name: "circular",
                        count: 1
                    }, {
                        name: "rotateLabel",
                        count: 1
                    }, {
                        name: "force",
                        count: 1
                    }, {
                        name: "initLayout",
                        count: 1
                    }, {
                        name: "repulsion",
                        count: 1
                    }, {
                        name: "gravity",
                        count: 1
                    }, {
                        name: "edgeLength",
                        count: 1
                    }, {
                        name: "layoutAnimation",
                        count: 1
                    }, {
                        name: "friction",
                        count: 1
                    }, {
                        name: "nodeScaleRatio",
                        count: 1
                    }, {
                        name: "edgeSymbol",
                        count: 1
                    }, {
                        name: "edgeSymbolSize",
                        count: 1
                    }, {
                        name: "autoCurveness",
                        count: 1
                    }, {
                        name: "fixed",
                        count: 1
                    }, {
                        name: "category",
                        count: 1
                    }, {
                        name: "ignoreForceLayout",
                        count: 1
                    }, {
                        name: "sankey",
                        count: 1
                    }, {
                        name: "nodeWidth",
                        count: 1
                    }, {
                        name: "nodeGap",
                        count: 1
                    }, {
                        name: "nodeAlign",
                        count: 1
                    }, {
                        name: "layoutIterations",
                        count: 1
                    }, {
                        name: "funnel",
                        count: 1
                    }, {
                        name: "minSize",
                        count: 1
                    }, {
                        name: "maxSize",
                        count: 1
                    }, {
                        name: "gap",
                        count: 1
                    }, {
                        name: "funnelAlign",
                        count: 1
                    }, {
                        name: "gauge",
                        count: 1
                    }, {
                        name: "overlap",
                        count: 1
                    }, {
                        name: "pointer",
                        count: 1
                    }, {
                        name: "anchor",
                        count: 1
                    }, {
                        name: "pictorialBar",
                        count: 1
                    }, {
                        name: "themeRiver",
                        count: 1
                    }, {
                        name: "singleAxisIndex",
                        count: 1
                    }, {
                        name: "date",
                        count: 1
                    }, {
                        name: "custom",
                        count: 1
                    }, {
                        name: "renderItem",
                        count: 1
                    }, {
                        name: "arguments",
                        count: 1
                    }, {
                        name: "params",
                        count: 1
                    }, {
                        name: "api",
                        count: 1
                    }, {
                        name: "styleEmphasis",
                        count: 1
                    }, {
                        name: "visual",
                        count: 1
                    }, {
                        name: "barLayout",
                        count: 1
                    }, {
                        name: "currentSeriesIndices",
                        count: 1
                    }, {
                        name: "getWidth",
                        count: 1
                    }, {
                        name: "getHeight",
                        count: 1
                    }, {
                        name: "getZr",
                        count: 1
                    }, {
                        name: "getDevicePixelRatio",
                        count: 1
                    }, {
                        name: "return",
                        count: 1
                    }, {
                        name: "return_group",
                        count: 1
                    }, {
                        name: "return_path",
                        count: 1
                    }, {
                        name: "pathData",
                        count: 1
                    }, {
                        name: "d",
                        count: 1
                    }, {
                        name: "return_image",
                        count: 1
                    }, {
                        name: "return_text",
                        count: 1
                    }, {
                        name: "return_rect",
                        count: 1
                    }, {
                        name: "return_circle",
                        count: 1
                    }, {
                        name: "return_ring",
                        count: 1
                    }, {
                        name: "return_sector",
                        count: 1
                    }, {
                        name: "return_arc",
                        count: 1
                    }, {
                        name: "return_polygon",
                        count: 1
                    }, {
                        name: "return_polyline",
                        count: 1
                    }, {
                        name: "return_line",
                        count: 1
                    }, {
                        name: "return_bezierCurve",
                        count: 1
                    }, {
                        name: "darkMode",
                        count: 1
                    }, {
                        name: "stateAnimation",
                        count: 1
                    }, {
                        name: "duration",
                        count: 1
                    }, {
                        name: "easing",
                        count: 1
                    }, {
                        name: "blendMode",
                        count: 1
                    }, {
                        name: "hoverLayerThreshold",
                        count: 1
                    }, {
                        name: "useUTC",
                        count: 1
                    }, {
                        name: "options",
                        count: 1
                    }, {
                        name: "media",
                        count: 1
                    }, {
                        name: "query",
                        count: 1
                    }, {
                        name: "minWidth",
                        count: 1
                    }, {
                        name: "maxHeight",
                        count: 1
                    }, {
                        name: "minAspectRatio",
                        count: 1
                    }],
                    nt = {};

                function at(t) {
                    return Promise.all(t.map((function(t) {
                        if ("string" == typeof t && (t = {
                                url: t,
                                type: t.match(/\.css$/) ? "css" : "js"
                            }), nt[t.url]) return nt[t.url];
                        var e = new Promise((function(e, n) {
                            if ("js" === t.type) {
                                var a = document.createElement("script");
                                a.src = t.url, a.async = !1, a.onload = function() {
                                    e()
                                }, a.onerror = function() {
                                    n()
                                }, document.body.appendChild(a)
                            } else if ("css" === t.type) {
                                var i = document.createElement("link");
                                i.rel = "stylesheet", i.href = t.url, i.onload = function() {
                                    e()
                                }, i.onerror = function() {
                                    n()
                                }, document.body.appendChild(i)
                            }
                        }));
                        return nt[t.url] = e, e
                    })))
                }
                var it = ["line", "bar", "pie", "scatter", "map", "candlestick", "radar", "boxplot", "heatmap", "graph", "lines", "tree", "treemap", "sunburst", "parallel", "sankey", "funnel", "gauge", "pictorialBar", "themeRiver", "calendar", "custom", "dataset", "dataZoom", "drag", "rich", "globe", "bar3D", "scatter3D", "surface", "map3D", "lines3D", "line3D", "scatterGL", "linesGL", "flowGL", "graphGL"],
                    ot = function(t) {
                        for (var e = {}, n = 0; n < t.length; n++) e[t[n]] = 1;
                        return location.href.indexOf("github.io") >= 0 ? {} : e
                    }(["effectScatter-map", "geo-lines", "geo-map-scatter", "heatmap-map", "lines-airline", "map-china", "map-china-dataRange", "map-labels", "map-locate", "map-province", "map-world", "map-world-dataRange", "scatter-map", "scatter-map-brush", "scatter-weibo", "scatter-world-population", "geo3d", "geo3d-with-different-height", "globe-country-carousel", "globe-with-echarts-surface", "map3d-alcohol-consumption", "map3d-wood-map", "scattergl-weibo"]),
                    rt = {};
                (location.search || "").substr(1).split("&").forEach((function(t) {
                    var e = t.split("=");
                    rt[e[0]] = e[1]
                }));
                var lt, st = ((lt = document.createElement("canvas")).width = lt.height = 1, !(!lt.getContext || !lt.getContext("2d")) && 0 === lt.toDataURL("image/webp").indexOf("data:image/webp")),
                    ct = {
                        localEChartsMinJS: "http://localhost/echarts/dist/echarts.js",
                        echartsMinJS: "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js",
                        echartsDir: "https://cdn.jsdelivr.net/npm/echarts@5",
                        echartsStatMinJS: "https://cdn.jsdelivr.net/npm/echarts-stat@latest/dist/ecStat.min.js",
                        echartsGLMinJS: "https://cdn.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js",
                        datGUIMinJS: "https://cdn.jsdelivr.net/npm/dat.gui@0.6.5/build/dat.gui.min.js",
                        monacoDir: "https://cdn.jsdelivr.net/npm/monaco-editor@0.21.2/min/vs",
                        aceDir: "https://cdn.jsdelivr.net/npm/ace-builds@1.4.12/src-min-noconflict"
                    },
                    ut = {
                        cdnRoot: "",
                        version: "",
                        locale: "",
                        darkMode: "dark" === rt.theme,
                        enableDecal: "decal" in rt,
                        renderer: rt.renderer || "canvas",
                        typeCheck: "monaco" === rt.editor,
                        useDirtyRect: "useDirtyRect" in rt,
                        runCode: "",
                        sourceCode: "",
                        runHash: "",
                        isMobile: window.innerWidth < 600,
                        editorStatus: {
                            type: "",
                            message: ""
                        }
                    };

                function dt() {
                    return new Promise((function(t) {
                        var e = rt.gl ? "data-gl" : "data";
                        $.ajax("".concat(ut.cdnRoot, "/").concat(e, "/").concat(rt.c, ".js?_v_").concat(ut.version), {
                            dataType: "text",
                            success: function(e) {
                                t(e)
                            }
                        })
                    }))
                }

                function pt(t) {
                    return t.replace(/\/\*[\w\W]*?\*\//, "").trim()
                }
                var ft = 123;

                function gt(t, e, n, a, i, o, r, l) {
                    var s, c = "function" == typeof t ? t.options : t;
                    if (e && (c.render = e, c.staticRenderFns = n, c._compiled = !0), a && (c.functional = !0), o && (c._scopeId = "data-v-" + o), r ? (s = function(t) {
                            (t = t || this.$vnode && this.$vnode.ssrContext || this.parent && this.parent.$vnode && this.parent.$vnode.ssrContext) || "undefined" == typeof __VUE_SSR_CONTEXT__ || (t = __VUE_SSR_CONTEXT__), i && i.call(this, t), t && t._registeredComponents && t._registeredComponents.add(r)
                        }, c._ssrRegister = s) : i && (s = l ? function() {
                            i.call(this, (c.functional ? this.parent : this).$root.$options.shadowRoot)
                        } : i), s)
                        if (c.functional) {
                            c._injectStyles = s;
                            var u = c.render;
                            c.render = function(t, e) {
                                return s.call(e), u(t, e)
                            }
                        } else {
                            var d = c.beforeCreate;
                            c.beforeCreate = d ? [].concat(d, s) : [s]
                        }
                    return {
                        exports: t,
                        options: c
                    }
                }
                var mt = gt({
                    props: ["initialCode"],
                    data: function() {
                        return {
                            shared: ut,
                            loading: !1
                        }
                    },
                    mounted: function() {
                        var t = this;
                        this.loading = !0, ("undefined" == typeof ace ? at([ct.aceDir + "/ace.js", ct.aceDir + "/ext-language_tools.js"]).then((function() {
                            var t = ace.require("ace/ext/language_tools"),
                                e = [];
                            et.forEach((function(t) {
                                e.push({
                                    caption: t.name,
                                    value: t.name,
                                    score: t.count,
                                    metal: "local"
                                })
                            })), t.addCompleter({
                                getCompletions: function(t, n, a, i, o) {
                                    o(null, e)
                                }
                            })
                        })) : Promise.resolve()).then((function() {
                            t.loading = !1;
                            var e = ace.edit(t.$el);
                            e.getSession().setMode("ace/mode/javascript"), e.setOptions({
                                enableBasicAutocompletion: !0,
                                enableSnippets: !0,
                                enableLiveAutocompletion: !0
                            }), t._editor = e, e.on("change", (function() {
                                ut.sourceCode = ut.runCode = e.getValue()
                            })), t.initialCode && t.setInitialCode(t.initialCode)
                        }))
                    },
                    methods: {
                        setInitialCode: function(t) {
                            this._editor && t && (this._editor.setValue(t || ""), this._editor.selection.setSelectionRange({
                                start: {
                                    row: 1,
                                    column: 4
                                },
                                end: {
                                    row: 1,
                                    column: 4
                                }
                            }))
                        }
                    },
                    watch: {
                        initialCode: function(t) {
                            this.setInitialCode(t)
                        }
                    }
                }, tt, [], !1, null, null, null);
                mt.options.__file = "src/editor/CodeAce.vue";
                const ht = mt.exports;
                var Ct = function() {
                    var t = this,
                        e = t.$createElement;
                    return (t._self._c || e)("div", {
                        directives: [{
                            name: "loading",
                            rawName: "v-loading",
                            value: t.loading,
                            expression: "loading"
                        }],
                        staticClass: "monaco-editor-main"
                    })
                };
                Ct._withStripped = !0;
                var yt = function() {
                    var t = this,
                        e = t.$createElement,
                        n = t._self._c || e;
                    return n("div", {
                        class: [t.inEditor && !t.shared.isMobile ? "" : "full"]
                    }, [n("div", {
                        directives: [{
                            name: "loading",
                            rawName: "v-loading",
                            value: t.loading,
                            expression: "loading"
                        }],
                        staticClass: "right-panel",
                        style: {
                            background: t.backgroundColor
                        },
                        attrs: {
                            id: "chart-panel"
                        }
                    }, [n("div", {
                        staticClass: "chart-container"
                    })]), t._v(" "), n("div", {
                        attrs: {
                            id: "tool-panel"
                        }
                    }, [n("div", {
                        staticClass: "left-panel"
                    }, [n("el-switch", {
                        staticClass: "dark-mode",
                        attrs: {
                            "active-color": "#181432",
                            "active-text": t.$t("editor.darkMode"),
                            "inactive-text": ""
                        },
                        model: {
                            value: t.shared.darkMode,
                            callback: function(e) {
                                t.$set(t.shared, "darkMode", e)
                            },
                            expression: "shared.darkMode"
                        }
                    }), t._v(" "), t.isGL ? t._e() : n("el-switch", {
                        staticClass: "enable-decal",
                        attrs: {
                            "active-text": t.$t("editor.enableDecal"),
                            "inactive-text": ""
                        },
                        model: {
                            value: t.shared.enableDecal,
                            callback: function(e) {
                                t.$set(t.shared, "enableDecal", e)
                            },
                            expression: "shared.enableDecal"
                        }
                    }), t._v(" "), t.isGL ? t._e() : n("el-popover", {
                        staticStyle: {
                            "margin-top": "3px"
                        },
                        attrs: {
                            placement: "bottom",
                            width: "450",
                            trigger: "click"
                        }
                    }, [n("div", {
                        staticClass: "render-config-container"
                    }, [n("el-row", {
                        attrs: {
                            gutter: 2,
                            type: "flex",
                            align: "middle"
                        }
                    }, [n("el-col", {
                        attrs: {
                            span: 12
                        }
                    }, [n("label", {
                        staticClass: "tool-label"
                    }, [t._v(t._s(t.$t("editor.renderer")))]), t._v(" "), n("el-radio-group", {
                        staticStyle: {
                            "text-transform": "uppercase"
                        },
                        attrs: {
                            size: "mini"
                        },
                        model: {
                            value: t.shared.renderer,
                            callback: function(e) {
                                t.$set(t.shared, "renderer", e)
                            },
                            expression: "shared.renderer"
                        }
                    }, [n("el-radio-button", {
                        attrs: {
                            label: "svg"
                        }
                    }), t._v(" "), n("el-radio-button", {
                        attrs: {
                            label: "canvas"
                        }
                    })], 1)], 1), t._v(" "), n("el-col", {
                        attrs: {
                            span: 12
                        }
                    }, ["canvas" === t.shared.renderer ? n("el-switch", {
                        attrs: {
                            "active-text": t.$t("editor.useDirtyRect"),
                            "inactive-text": ""
                        },
                        model: {
                            value: t.shared.useDirtyRect,
                            callback: function(e) {
                                t.$set(t.shared, "useDirtyRect", e)
                            },
                            expression: "shared.useDirtyRect"
                        }
                    }) : t._e()], 1)], 1)], 1), t._v(" "), n("span", {
                        staticClass: "render-config-trigger",
                        attrs: {
                            slot: "reference"
                        },
                        slot: "reference"
                    }, [n("i", {
                        staticClass: "el-icon-setting el-icon--left"
                    }), t._v(t._s(t.$t("editor.renderCfgTitle")))])])], 1), t._v(" "), t.inEditor ? [t.shared.isMobile ? t._e() : n("button", {
                        staticClass: "download btn btn-sm",
                        on: {
                            click: t.downloadExample
                        }
                    }, [t._v(t._s(t.$t("editor.download")))]), t._v(" "), n("a", {
                        staticClass: "screenshot",
                        attrs: {
                            target: "_blank"
                        },
                        on: {
                            click: t.screenshot
                        }
                    }, [n("i", {
                        staticClass: "el-icon-camera-solid"
                    })])] : n("a", {
                        staticClass: "edit btn btn-sm",
                        attrs: {
                            href: t.editLink,
                            target: "_blank"
                        }
                    }, [t._v(t._s(t.$t("editor.edit")))])], 2)])
                };
                yt._withStripped = !0;
                var vt = function() {
                    function t(t) {
                        var e = this.dom = document.createElement("div");
                        for (var n in e.className = "ec-debug-dirty-rect", t = Object.assign({}, t), Object.assign(t, {
                                backgroundColor: "rgba(0, 0, 255, 0.2)",
                                border: "1px solid #00f"
                            }), e.style.cssText = "\nposition: absolute;\nopacity: 0;\ntransition: opacity 0.5s linear;\npointer-events: none;\n", t) t.hasOwnProperty(n) && (e.style[n] = t[n])
                    }
                    return t.prototype.update = function(t) {
                        var e = this.dom.style;
                        e.width = t.width + "px", e.height = t.height + "px", e.left = t.x + "px", e.top = t.y + "px"
                    }, t.prototype.hide = function() {
                        this.dom.style.opacity = "0"
                    }, t.prototype.show = function() {
                        var t = this;
                        clearTimeout(this._hideTimeout), this.dom.style.opacity = "1", this._hideTimeout = setTimeout((function() {
                            t.hide()
                        }), 500)
                    }, t
                }();

                function bt(t) {
                    return (bt = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(t) {
                        return typeof t
                    } : function(t) {
                        return t && "function" == typeof Symbol && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t
                    })(t)
                }
                var _t = n(279),
                    Lt = n.n(_t);
                let wt = null,
                    xt = null;

                function kt(t, e = {}) {
                    let n = document.createElement(t);
                    return Object.keys(e).forEach((t => {
                        n[t] = e[t]
                    })), n
                }

                function St(t, e, n) {
                    return (window.getComputedStyle(t, n || null) || {
                        display: "none"
                    })[e]
                }

                function Nt(t) {
                    if (!document.documentElement.contains(t)) return {
                        detached: !0,
                        rendered: !1
                    };
                    let e = t;
                    for (; e !== document;) {
                        if ("none" === St(e, "display")) return {
                            detached: !1,
                            rendered: !1
                        };
                        e = e.parentNode
                    }
                    return {
                        detached: !1,
                        rendered: !0
                    }
                }
                let Mt = 0,
                    Tt = null;

                function Ot(t, e) {
                    if (t.__resize_mutation_handler__ || (t.__resize_mutation_handler__ = Dt.bind(t)), !t.__resize_listeners__)
                        if (t.__resize_listeners__ = [], window.ResizeObserver) {
                            let {
                                offsetWidth: e,
                                offsetHeight: n
                            } = t, a = new ResizeObserver((() => {
                                (t.__resize_observer_triggered__ || (t.__resize_observer_triggered__ = !0, t.offsetWidth !== e || t.offsetHeight !== n)) && Et(t)
                            })), {
                                detached: i,
                                rendered: o
                            } = Nt(t);
                            t.__resize_observer_triggered__ = !1 === i && !1 === o, t.__resize_observer__ = a, a.observe(t)
                        } else if (t.attachEvent && t.addEventListener) t.__resize_legacy_resize_handler__ = function() {
                        Et(t)
                    }, t.attachEvent("onresize", t.__resize_legacy_resize_handler__), document.addEventListener("DOMSubtreeModified", t.__resize_mutation_handler__);
                    else if (Mt || (Tt = function(t) {
                            var e = document.createElement("style");
                            return e.type = "text/css", e.styleSheet ? e.styleSheet.cssText = t : e.appendChild(document.createTextNode(t)), (document.querySelector("head") || document.body).appendChild(e), e
                        }('.resize-triggers{visibility:hidden;opacity:0;pointer-events:none}.resize-contract-trigger,.resize-contract-trigger:before,.resize-expand-trigger,.resize-triggers{content:"";position:absolute;top:0;left:0;height:100%;width:100%;overflow:hidden}.resize-contract-trigger,.resize-expand-trigger{background:#eee;overflow:auto}.resize-contract-trigger:before{width:200%;height:200%}')), function(t) {
                            let e = St(t, "position");
                            e && "static" !== e || (t.style.position = "relative"), t.__resize_old_position__ = e, t.__resize_last__ = {};
                            let n = kt("div", {
                                    className: "resize-triggers"
                                }),
                                a = kt("div", {
                                    className: "resize-expand-trigger"
                                }),
                                i = kt("div"),
                                o = kt("div", {
                                    className: "resize-contract-trigger"
                                });
                            a.appendChild(i), n.appendChild(a), n.appendChild(o), t.appendChild(n), t.__resize_triggers__ = {
                                triggers: n,
                                expand: a,
                                expandChild: i,
                                contract: o
                            }, Pt(t), t.addEventListener("scroll", At, !0), t.__resize_last__ = {
                                width: t.offsetWidth,
                                height: t.offsetHeight
                            }
                        }(t), t.__resize_rendered__ = Nt(t).rendered, window.MutationObserver) {
                        let e = new MutationObserver(t.__resize_mutation_handler__);
                        e.observe(document, {
                            attributes: !0,
                            childList: !0,
                            characterData: !0,
                            subtree: !0
                        }), t.__resize_mutation_observer__ = e
                    }
                    t.__resize_listeners__.push(e), Mt++
                }

                function Dt() {
                    let {
                        rendered: t,
                        detached: e
                    } = Nt(this);
                    t !== this.__resize_rendered__ && (!e && this.__resize_triggers__ && (Pt(this), this.addEventListener("scroll", At, !0)), this.__resize_rendered__ = t, Et(this))
                }

                function At() {
                    var t, e;
                    Pt(this), this.__resize_raf__ && (t = this.__resize_raf__, xt || (xt = (window.cancelAnimationFrame || window.webkitCancelAnimationFrame || window.mozCancelAnimationFrame || function(t) {
                        clearTimeout(t)
                    }).bind(window)), xt(t)), this.__resize_raf__ = (e = () => {
                        let t = function(t) {
                            let {
                                width: e,
                                height: n
                            } = t.__resize_last__, {
                                offsetWidth: a,
                                offsetHeight: i
                            } = t;
                            return a !== e || i !== n ? {
                                width: a,
                                height: i
                            } : null
                        }(this);
                        t && (this.__resize_last__ = t, Et(this))
                    }, wt || (wt = (window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || function(t) {
                        return setTimeout(t, 16)
                    }).bind(window)), wt(e))
                }

                function Et(t) {
                    t && t.__resize_listeners__ && t.__resize_listeners__.forEach((e => {
                        e.call(t, t)
                    }))
                }

                function Pt(t) {
                    let {
                        expand: e,
                        expandChild: n,
                        contract: a
                    } = t.__resize_triggers__, {
                        scrollWidth: i,
                        scrollHeight: o
                    } = a, {
                        offsetWidth: r,
                        offsetHeight: l,
                        scrollWidth: s,
                        scrollHeight: c
                    } = e;
                    a.scrollLeft = i, a.scrollTop = o, n.style.width = r + 1 + "px", n.style.height = l + 1 + "px", e.scrollLeft = s, e.scrollTop = c
                }
                const Ft = [{
                        category: ["bar"],
                        id: "bar-background",
                        tags: [],
                        title: "Bar with Background",
                        titleCN: "带背景色的柱状图",
                        difficulty: 0
                    }, {
                        category: ["custom"],
                        id: "bar-histogram",
                        tags: [],
                        title: "Histogram with Custom Series",
                        titleCN: "直方图（自定义系列）",
                        difficulty: 0
                    }, {
                        category: ["bar"],
                        id: "bar-simple",
                        tags: [],
                        title: "Basic Bar",
                        titleCN: "基础柱状图",
                        difficulty: 0
                    }, {
                        category: ["bar"],
                        id: "bar-tick-align",
                        tags: [],
                        title: "Axis Align with Tick",
                        titleCN: "坐标轴刻度与标签对齐",
                        difficulty: 0
                    }, {
                        category: ["calendar"],
                        id: "calendar-simple",
                        tags: [],
                        title: "Simple Calendar",
                        titleCN: "基础日历图",
                        difficulty: 0
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-simple",
                        tags: [],
                        title: "Basic Candlestick",
                        titleCN: "基础 K 线图",
                        difficulty: 0
                    }, {
                        category: ["dataset", "bar", "transform"],
                        id: "data-transform-sort-bar",
                        tags: [],
                        title: "Sort Data in Bar Chart",
                        titleCN: "柱状图排序",
                        difficulty: 0
                    }, {
                        category: ["heatmap"],
                        id: "heatmap-cartesian",
                        tags: [],
                        title: "Heatmap on Cartesian",
                        titleCN: "笛卡尔坐标系上的热力图",
                        difficulty: 0
                    }, {
                        category: ["line"],
                        id: "line-simple",
                        tags: [],
                        title: "Basic Line Chart",
                        titleCN: "基础折线图",
                        difficulty: 0
                    }, {
                        category: ["line"],
                        id: "line-smooth",
                        tags: [],
                        title: "Smoothed Line Chart",
                        titleCN: "基础平滑折线图",
                        difficulty: 0
                    }, {
                        category: ["pie"],
                        id: "pie-simple",
                        tags: [],
                        title: "Referer of a website",
                        titleCN: "某站点用户访问来源",
                        difficulty: 0
                    }, {
                        category: ["radar"],
                        id: "radar",
                        tags: [],
                        title: "Basic Radar Chart",
                        titleCN: "基础雷达图",
                        difficulty: 0
                    }, {
                        category: ["sankey"],
                        id: "sankey-simple",
                        tags: [],
                        title: "Basic Sankey",
                        titleCN: "基础桑基图",
                        difficulty: 0
                    }, {
                        category: ["scatter"],
                        id: "scatter-simple",
                        tags: [],
                        title: "Basic Scatter Chart",
                        titleCN: "基础散点图",
                        difficulty: 0
                    }, {
                        category: ["line"],
                        id: "area-basic",
                        tags: [],
                        title: "Basic area chart",
                        titleCN: "基础面积图",
                        difficulty: 1
                    }, {
                        category: ["bar"],
                        id: "bar-data-color",
                        tags: [],
                        title: "Set Style of Single Bar.",
                        titleCN: "自定义单个柱子颜色",
                        difficulty: 1
                    }, {
                        category: ["bar"],
                        id: "bar-waterfall",
                        tags: [],
                        title: "Waterfall Chart",
                        titleCN: "瀑布图（柱状图模拟）",
                        difficulty: 1
                    }, {
                        category: ["calendar", "heatmap"],
                        id: "calendar-heatmap",
                        tags: [],
                        title: "Calendar Heatmap",
                        titleCN: "日历热力图",
                        difficulty: 1
                    }, {
                        category: ["calendar", "heatmap"],
                        id: "calendar-vertical",
                        tags: [],
                        title: "Calendar Heatmap Vertical",
                        titleCN: "纵向日历图",
                        difficulty: 1
                    }, {
                        category: ["candlestick"],
                        id: "custom-ohlc",
                        tags: [],
                        title: "OHLC Chart",
                        titleCN: "OHLC 图（使用自定义系列）",
                        difficulty: 1
                    }, {
                        category: ["custom"],
                        id: "custom-profit",
                        tags: [],
                        title: "Profit",
                        titleCN: "利润分布直方图",
                        difficulty: 1
                    }, {
                        category: ["dataset", "bar"],
                        id: "dataset-encode0",
                        tags: [],
                        title: "Simple Encode",
                        titleCN: "指定数据到坐标轴的映射",
                        difficulty: 1
                    }, {
                        category: ["gauge"],
                        id: "gauge",
                        tags: [],
                        title: "Gauge Basic chart",
                        titleCN: "基础仪表盘",
                        difficulty: 1
                    }, {
                        category: ["gauge"],
                        id: "gauge-simple",
                        tags: [],
                        title: "Simple Gauge",
                        titleCN: "带标签数字动画的基础仪表盘",
                        difficulty: 1
                    }, {
                        category: ["graph"],
                        id: "graph-force2",
                        tags: [],
                        title: "Force Layout",
                        titleCN: "力引导布局",
                        difficulty: 1
                    }, {
                        category: ["line"],
                        id: "line-stack",
                        tags: [],
                        title: "Stacked Line Chart",
                        titleCN: "折线图堆叠",
                        difficulty: 1
                    }, {
                        category: ["parallel"],
                        id: "parallel-simple",
                        tags: [],
                        title: "Basic Parallel",
                        titleCN: "基础平行坐标",
                        difficulty: 1
                    }, {
                        category: ["pie"],
                        id: "pie-borderRadius",
                        tags: [],
                        title: "Doughnut Chart with Rounded Corner",
                        titleCN: "圆角环形图",
                        difficulty: 1
                    }, {
                        category: ["pie"],
                        id: "pie-doughnut",
                        tags: [],
                        title: "Doughnut Chart",
                        titleCN: "环形图",
                        difficulty: 1
                    }, {
                        category: ["radar"],
                        id: "radar-aqi",
                        tags: [],
                        title: "AQI - Radar Chart",
                        titleCN: "AQI - 雷达图",
                        difficulty: 1
                    }, {
                        category: ["sankey"],
                        id: "sankey-vertical",
                        tags: [],
                        title: "Sankey Orient Vertical",
                        titleCN: "垂直方向的桑基图",
                        difficulty: 1
                    }, {
                        category: ["scatter"],
                        id: "scatter-anscombe-quartet",
                        tags: [],
                        title: "Anscomb's quartet",
                        titleCN: "Anscomb's quartet",
                        difficulty: 1
                    }, {
                        category: ["scatter"],
                        id: "scatter-clustering",
                        tags: [],
                        title: "Clustering Process",
                        titleCN: "数据聚合",
                        difficulty: 1
                    }, {
                        category: ["scatter"],
                        id: "scatter-clustering-process",
                        tags: [],
                        title: "Clustering Process",
                        titleCN: "聚合过程可视化",
                        difficulty: 1
                    }, {
                        category: ["scatter"],
                        id: "scatter-exponential-regression",
                        tags: [],
                        title: "Exponential Regression",
                        titleCN: "指数回归（使用统计插件）",
                        difficulty: 1
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-simple",
                        tags: [],
                        title: "Basic Sunburst",
                        titleCN: "基础旭日图",
                        difficulty: 1
                    }, {
                        category: ["line"],
                        id: "area-stack",
                        tags: [],
                        title: "Stacked area chart",
                        titleCN: "堆叠面积图",
                        difficulty: 2
                    }, {
                        category: ["line"],
                        id: "area-stack-gradient",
                        tags: [],
                        title: "Gradient Stacked area chart",
                        titleCN: "渐变堆叠面积图",
                        difficulty: 2
                    }, {
                        category: ["bar"],
                        id: "bar-negative2",
                        tags: [],
                        title: "Bar Chart with Negative Value",
                        titleCN: "交错正负轴标签",
                        difficulty: 2
                    }, {
                        category: ["bar"],
                        id: "bar-y-category",
                        tags: [],
                        title: "World Total Population",
                        titleCN: "世界人口总量 - 条形图",
                        difficulty: 2
                    }, {
                        category: ["calendar"],
                        id: "calendar-horizontal",
                        tags: [],
                        title: "Calendar Heatmap Horizontal",
                        titleCN: "横向日力图",
                        difficulty: 2
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-sh",
                        tags: [],
                        title: "ShangHai Index",
                        titleCN: "上证指数",
                        difficulty: 2
                    }, {
                        category: ["custom", "dataZoom"],
                        id: "custom-error-scatter",
                        tags: [],
                        title: "Error Scatter on Catesian",
                        titleCN: "使用自定系列给散点图添加误差范围",
                        difficulty: 2
                    }, {
                        category: ["scatter"],
                        id: "effectScatter-map",
                        tags: [],
                        title: "Air Quality",
                        titleCN: "全国主要城市空气质量",
                        difficulty: 2
                    }, {
                        category: ["gauge"],
                        id: "gauge-speed",
                        tags: [],
                        title: "Speed Gauge",
                        titleCN: "速度仪表盘",
                        difficulty: 2
                    }, {
                        category: ["graph"],
                        id: "graph-grid",
                        tags: [],
                        title: "Graph on Cartesian",
                        titleCN: "笛卡尔坐标系上的 Graph",
                        difficulty: 2
                    }, {
                        category: ["graph"],
                        id: "graph-simple",
                        tags: [],
                        title: "Simple Graph",
                        titleCN: "Graph 简单示例",
                        difficulty: 2
                    }, {
                        category: ["heatmap"],
                        id: "heatmap-large",
                        tags: [],
                        title: "Heatmap - 2w data",
                        titleCN: "热力图 - 2w 数据",
                        difficulty: 2
                    }, {
                        category: ["heatmap"],
                        id: "heatmap-large-piecewise",
                        tags: [],
                        title: "Heatmap - Discrete Mapping of Color",
                        titleCN: "热力图 - 颜色的离散映射",
                        difficulty: 2
                    }, {
                        category: ["line"],
                        id: "line-marker",
                        tags: [],
                        title: "Temperature Change in the coming week",
                        titleCN: "未来一周气温变化",
                        difficulty: 2
                    }, {
                        category: ["parallel"],
                        id: "parallel-aqi",
                        tags: [],
                        title: "Parallel Aqi",
                        titleCN: "AQI 分布（平行坐标）",
                        difficulty: 2
                    }, {
                        category: ["pie"],
                        id: "pie-custom",
                        tags: [],
                        title: "Customized Pie",
                        titleCN: "饼图自定义样式",
                        difficulty: 2
                    }, {
                        category: ["pie"],
                        id: "pie-pattern",
                        tags: [],
                        title: "Texture on Pie Chart",
                        titleCN: "饼图纹理",
                        difficulty: 2
                    }, {
                        category: ["pie"],
                        id: "pie-roseType",
                        tags: [],
                        title: "Nightingale's Rose Diagram",
                        titleCN: "南丁格尔玫瑰图",
                        difficulty: 2
                    }, {
                        category: ["pie"],
                        id: "pie-roseType-simple",
                        tags: [],
                        title: "Nightingale's Rose Diagram",
                        titleCN: "基础南丁格尔玫瑰图",
                        difficulty: 2
                    }, {
                        category: ["radar"],
                        id: "radar-custom",
                        tags: [],
                        title: "Customized Radar Chart",
                        titleCN: "自定义雷达图",
                        difficulty: 2
                    }, {
                        category: ["sankey"],
                        id: "sankey-itemstyle",
                        tags: [],
                        title: "Specify ItemStyle for Each Node in Sankey",
                        titleCN: "桑基图节点自定义样式",
                        difficulty: 2
                    }, {
                        category: ["sankey"],
                        id: "sankey-levels",
                        tags: [],
                        title: "Sankey with Levels Setting",
                        titleCN: "桑基图层级自定义样式",
                        difficulty: 2
                    }, {
                        category: ["scatter"],
                        id: "scatter-effect",
                        tags: [],
                        title: "Effect Scatter Chart",
                        titleCN: "涟漪特效散点图",
                        difficulty: 2
                    }, {
                        category: ["scatter"],
                        id: "scatter-linear-regression",
                        tags: [],
                        title: "Linear Regression",
                        titleCN: "线性回归（使用统计插件）",
                        difficulty: 2
                    }, {
                        category: ["scatter"],
                        id: "scatter-polynomial-regression",
                        tags: [],
                        title: "Polynomial Regression",
                        titleCN: "多项式回归（使用统计插件）",
                        difficulty: 2
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-borderRadius",
                        tags: [],
                        title: "Sunburst with Rounded Corner",
                        titleCN: "圆角旭日图",
                        difficulty: 2
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-label-rotate",
                        tags: [],
                        title: "Sunburst Label Rotate",
                        titleCN: "旭日图标签旋转",
                        difficulty: 2
                    }, {
                        category: ["line", "visualMap"],
                        id: "area-pieces",
                        tags: [],
                        title: "Area Pieces",
                        titleCN: "折线图区域高亮",
                        difficulty: 3
                    }, {
                        category: ["bar"],
                        id: "bar-gradient",
                        tags: [],
                        title: "Clickable Column Chart with Gradient",
                        titleCN: "特性示例：渐变色 阴影 点击缩放",
                        difficulty: 3
                    }, {
                        category: ["bar"],
                        id: "bar-label-rotation",
                        tags: [],
                        title: "Bar Label Rotation",
                        titleCN: "柱状图标签旋转",
                        difficulty: 3
                    }, {
                        category: ["bar"],
                        id: "bar-stack",
                        tags: [],
                        title: "Stacked Column Chart",
                        titleCN: "堆叠柱状图",
                        difficulty: 3
                    }, {
                        category: ["bar"],
                        id: "bar-waterfall2",
                        tags: [],
                        title: "Waterfall Chart",
                        titleCN: "阶梯瀑布图（柱状图模拟）",
                        difficulty: 3
                    }, {
                        category: ["bar"],
                        id: "bar-y-category-stack",
                        tags: [],
                        title: "Stacked Horizontal Bar",
                        titleCN: "堆叠条形图",
                        difficulty: 3
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-large",
                        tags: [],
                        title: "Large Scale Candlestick",
                        titleCN: "大数据量K线图",
                        difficulty: 3
                    }, {
                        category: ["custom"],
                        id: "custom-bar-trend",
                        tags: [],
                        title: "Custom Bar Trend",
                        titleCN: "使用自定义系列添加柱状图趋势",
                        difficulty: 3
                    }, {
                        category: ["custom"],
                        id: "custom-cartesian-polygon",
                        tags: [],
                        title: "Custom Cartesian Polygon",
                        titleCN: "自定义多边形图",
                        difficulty: 3
                    }, {
                        category: ["custom"],
                        id: "custom-error-bar",
                        tags: [],
                        title: "Error Bar on Catesian",
                        titleCN: "使用自定系列给柱状图添加误差范围",
                        difficulty: 3
                    }, {
                        category: ["custom"],
                        id: "custom-profile",
                        tags: [],
                        title: "Profile",
                        titleCN: "性能分析图",
                        difficulty: 3
                    }, {
                        category: ["custom"],
                        id: "cycle-plot",
                        tags: [],
                        title: "Cycle Plot",
                        titleCN: "Cycle Plot",
                        difficulty: 3
                    }, {
                        category: ["line"],
                        id: "data-transform-filter",
                        tags: [],
                        title: "Data Transform Fitler",
                        titleCN: "数据过滤",
                        difficulty: 3
                    }, {
                        category: ["dataset", "pie", "transform"],
                        id: "data-transform-multiple-pie",
                        tags: [],
                        title: "Partition Data to Pies",
                        titleCN: "分割数据到数个饼图",
                        difficulty: 3
                    }, {
                        category: ["dataset", "pie"],
                        id: "dataset-default",
                        tags: [],
                        title: "Default arrangement",
                        titleCN: "默认 encode 设置",
                        difficulty: 3
                    }, {
                        category: ["dataset"],
                        id: "dataset-encode1",
                        tags: [],
                        title: "Encode and Matrix",
                        titleCN: "指定数据到坐标轴的映射",
                        difficulty: 3
                    }, {
                        category: ["gauge"],
                        id: "gauge-progress",
                        tags: [],
                        title: "Grogress Gauge",
                        titleCN: "进度仪表盘",
                        difficulty: 3
                    }, {
                        category: ["gauge"],
                        id: "gauge-stage",
                        tags: [],
                        title: "Stage Speed Gauge",
                        titleCN: "阶段速度仪表盘",
                        difficulty: 3
                    }, {
                        category: ["graph"],
                        id: "graph-force",
                        tags: [],
                        title: "Force Layout",
                        titleCN: "力引导布局",
                        difficulty: 3
                    }, {
                        category: ["graph"],
                        id: "graph-label-overlap",
                        tags: [],
                        title: "Hide Overlapped Label",
                        titleCN: "关系图自动隐藏重叠标签",
                        difficulty: 3
                    }, {
                        category: ["heatmap"],
                        id: "heatmap-bmap",
                        tags: ["bmap"],
                        title: "Heatmap on Baidu Map Extension",
                        titleCN: "热力图与百度地图扩展",
                        difficulty: 3
                    }, {
                        category: ["heatmap"],
                        id: "heatmap-map",
                        tags: [],
                        title: "Air Qulity",
                        titleCN: "全国主要城市空气质量",
                        difficulty: 3
                    }, {
                        category: ["line"],
                        id: "line-gradient",
                        tags: [],
                        title: "Line Gradient",
                        titleCN: "折线图的渐变",
                        difficulty: 3
                    }, {
                        category: ["line"],
                        id: "line-sections",
                        tags: [],
                        title: "Distribution of Electricity",
                        titleCN: "一天用电量分布",
                        difficulty: 3
                    }, {
                        category: ["pie"],
                        id: "pie-alignTo",
                        tags: [],
                        title: "Pie Label Align",
                        titleCN: "饼图标签对齐",
                        difficulty: 3
                    }, {
                        category: ["pie"],
                        id: "pie-labelLine-adjust",
                        tags: [],
                        title: "Label Line Adjust",
                        titleCN: "饼图引导线调整",
                        difficulty: 3
                    }, {
                        category: ["radar"],
                        id: "radar2",
                        tags: [],
                        title: "Proportion of Browsers",
                        titleCN: "浏览器占比变化",
                        difficulty: 3
                    }, {
                        category: ["sankey"],
                        id: "sankey-energy",
                        tags: [],
                        title: "Gradient Edge",
                        titleCN: "桑基图渐变色边",
                        difficulty: 3
                    }, {
                        category: ["sankey"],
                        id: "sankey-nodeAlign-left",
                        tags: [],
                        title: "Node Align Left in Sankey",
                        titleCN: "桑基图左对齐布局",
                        difficulty: 3
                    }, {
                        category: ["sankey"],
                        id: "sankey-nodeAlign-right",
                        tags: [],
                        title: "Node Align Right in Sankey",
                        titleCN: "桑基图右对齐布局",
                        difficulty: 3
                    }, {
                        category: ["scatter"],
                        id: "scatter-punchCard",
                        tags: [],
                        title: "Punch Card of Github",
                        titleCN: "GitHub 打卡气泡图",
                        difficulty: 3
                    }, {
                        category: ["scatter"],
                        id: "scatter-single-axis",
                        tags: [],
                        title: "Scatter on Single Axis",
                        titleCN: "单轴散点图",
                        difficulty: 3
                    }, {
                        category: ["scatter"],
                        id: "scatter-weight",
                        tags: [],
                        title: "Distribution of Height and Weight",
                        titleCN: "男性女性身高体重分布",
                        difficulty: 3
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-monochrome",
                        tags: [],
                        title: "Monochrome Sunburst",
                        titleCN: "Monochrome Sunburst",
                        difficulty: 3
                    }, {
                        category: ["line", "dataZoom"],
                        id: "area-simple",
                        tags: [],
                        title: "Large scale area chart",
                        titleCN: "大数据量面积图",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "bar-brush",
                        tags: [],
                        title: "Brush Select on Column Chart",
                        titleCN: "柱状图框选",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "bar-negative",
                        tags: [],
                        title: "Bar Chart with Negative Value",
                        titleCN: "正负条形图",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "bar1",
                        tags: [],
                        title: "Rainfall and Evaporation",
                        titleCN: "某地区蒸发量和降水量",
                        difficulty: 4
                    }, {
                        category: ["calendar", "graph"],
                        id: "calendar-graph",
                        tags: [],
                        title: "Calendar Graph",
                        titleCN: "日历关系图",
                        difficulty: 4
                    }, {
                        category: ["calendar"],
                        id: "calendar-lunar",
                        tags: [],
                        title: "Calendar Lunar",
                        titleCN: "农历日历图",
                        difficulty: 4
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-touch",
                        tags: [],
                        title: "Axis Pointer Link and Touch",
                        titleCN: "触屏上的坐标轴指示器",
                        difficulty: 4
                    }, {
                        category: ["line"],
                        id: "confidence-band",
                        tags: [],
                        title: "Confidence Band",
                        titleCN: "Confidence Band",
                        difficulty: 4
                    }, {
                        category: ["custom", "dataZoom", "drag"],
                        id: "custom-gantt-flight",
                        tags: [],
                        title: "Gantt Chart of Airport Flights",
                        titleCN: "机场航班甘特图",
                        difficulty: 4
                    }, {
                        category: ["custom"],
                        id: "custom-polar-heatmap",
                        tags: [],
                        title: "Polar Heatmap",
                        titleCN: "极坐标热力图（自定义系列）",
                        difficulty: 4
                    }, {
                        category: ["boxplot"],
                        id: "data-transform-aggregate",
                        tags: [],
                        title: "Data Transform Simple Aggregate",
                        titleCN: "简单的数据聚合",
                        difficulty: 4
                    }, {
                        category: ["gauge"],
                        id: "gauge-grade",
                        tags: [],
                        title: "Grade Gauge",
                        titleCN: "等级仪表盘",
                        difficulty: 4
                    }, {
                        category: ["gauge"],
                        id: "gauge-multi-title",
                        tags: [],
                        title: "Multi Title Gauge",
                        titleCN: "多标题仪表盘",
                        difficulty: 4
                    }, {
                        category: ["gauge"],
                        id: "gauge-temperature",
                        tags: [],
                        title: "Temperature Gauge chart",
                        titleCN: "气温仪表盘",
                        difficulty: 4
                    }, {
                        category: ["graph"],
                        id: "graph",
                        tags: [],
                        title: "Les Miserables",
                        titleCN: "悲惨世界人物关系图",
                        difficulty: 4
                    }, {
                        category: ["line"],
                        id: "grid-multiple",
                        tags: [],
                        title: "Rainfall and Water Flow",
                        titleCN: "雨量流量关系图",
                        difficulty: 4
                    }, {
                        category: ["line"],
                        id: "line-aqi",
                        tags: [],
                        title: "Beijing AQI",
                        titleCN: "北京 AQI 可视化",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "mix-line-bar",
                        tags: [],
                        title: "Mixed Line and Bar",
                        titleCN: "折柱混合",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "mix-zoom-on-value",
                        tags: [],
                        title: "Mix Zoom On Value",
                        titleCN: "多数值轴轴缩放",
                        difficulty: 4
                    }, {
                        category: ["line"],
                        id: "multiple-x-axis",
                        tags: [],
                        title: "Multiple X Axes",
                        titleCN: "多 X 轴",
                        difficulty: 4
                    }, {
                        category: ["bar"],
                        id: "multiple-y-axis",
                        tags: [],
                        title: "Multiple Y Axes",
                        titleCN: "多 Y 轴示例",
                        difficulty: 4
                    }, {
                        category: ["parallel"],
                        id: "parallel-nutrients",
                        tags: [],
                        title: "Parallel Nutrients",
                        titleCN: "营养结构（平行坐标）",
                        difficulty: 4
                    }, {
                        category: ["pie"],
                        id: "pie-legend",
                        tags: [],
                        title: "Pie with Scrollable Legend",
                        titleCN: "可滚动的图例",
                        difficulty: 4
                    }, {
                        category: ["pie", "rich"],
                        id: "pie-rich-text",
                        tags: [],
                        title: "Pie Special Label",
                        titleCN: "富文本标签",
                        difficulty: 4
                    }, {
                        category: ["scatter"],
                        id: "scatter-label-align-right",
                        tags: [],
                        title: "Align Label on the Top",
                        titleCN: "散点图标签顶部对齐",
                        difficulty: 4
                    }, {
                        category: ["scatter"],
                        id: "scatter-label-align-top",
                        tags: [],
                        title: "Align Label on the Top",
                        titleCN: "散点图标签顶部对齐",
                        difficulty: 4
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-visualMap",
                        tags: [],
                        title: "Sunburst VisualMap",
                        titleCN: "旭日图使用视觉编码",
                        difficulty: 4
                    }, {
                        category: ["line"],
                        id: "area-rainfall",
                        tags: [],
                        title: "Rainfall",
                        titleCN: "雨量流量关系图",
                        difficulty: 5
                    }, {
                        category: ["line"],
                        id: "area-time-axis",
                        tags: [],
                        title: "Area Chart with Time Axis",
                        titleCN: "时间轴折线图",
                        difficulty: 5
                    }, {
                        category: ["bar"],
                        id: "bar-animation-delay",
                        tags: [],
                        title: "Animation Delay",
                        titleCN: "柱状图动画延迟",
                        difficulty: 5
                    }, {
                        category: ["bar"],
                        id: "bar-large",
                        tags: [],
                        title: "Large Scale Bar Chart",
                        titleCN: "大数据量柱图",
                        difficulty: 5
                    }, {
                        category: ["bar"],
                        id: "bar-race",
                        tags: [],
                        title: "Bar Race",
                        titleCN: "动态排序柱状图",
                        difficulty: 5
                    }, {
                        category: ["dataset", "line", "pie"],
                        id: "dataset-link",
                        tags: [],
                        title: "Share Dataset",
                        titleCN: "联动和共享数据集",
                        difficulty: 5
                    }, {
                        category: ["dataset", "bar"],
                        id: "dataset-series-layout-by",
                        tags: [],
                        title: "Series Layout By Column or Row",
                        titleCN: "系列按行和按列排布",
                        difficulty: 5
                    }, {
                        category: ["dataset", "bar"],
                        id: "dataset-simple0",
                        tags: [],
                        title: "Simple Example of Dataset",
                        titleCN: "最简单的数据集（dataset）",
                        difficulty: 5
                    }, {
                        category: ["dataset", "bar"],
                        id: "dataset-simple1",
                        tags: [],
                        title: "Dataset in Object Array",
                        titleCN: "对象数组的输入格式",
                        difficulty: 5
                    }, {
                        category: ["line"],
                        id: "dynamic-data2",
                        tags: [],
                        title: "Dynamic Data + Time Axis",
                        titleCN: "动态数据 + 时间坐标轴",
                        difficulty: 5
                    }, {
                        category: ["gauge"],
                        id: "gauge-ring",
                        tags: [],
                        title: "Ring Gauge",
                        titleCN: "得分环",
                        difficulty: 5
                    }, {
                        category: ["graph"],
                        id: "graph-circular-layout",
                        tags: [],
                        title: "Les Miserables",
                        titleCN: "悲惨世界人物关系图(环形布局)",
                        difficulty: 5
                    }, {
                        category: ["line"],
                        id: "line-function",
                        tags: [],
                        title: "Function Plot",
                        titleCN: "函数绘图",
                        difficulty: 5
                    }, {
                        category: ["line"],
                        id: "line-race",
                        tags: [],
                        title: "Line Race",
                        titleCN: "动态排序折线图",
                        difficulty: 5
                    }, {
                        category: ["pie", "rich"],
                        id: "pie-nest",
                        tags: [],
                        title: "Nested Pies",
                        titleCN: "嵌套环形图",
                        difficulty: 5
                    }, {
                        category: ["scatter"],
                        id: "scatter-large",
                        tags: [],
                        title: "Large Scatter",
                        titleCN: "大规模散点图",
                        difficulty: 5
                    }, {
                        category: ["scatter"],
                        id: "scatter-nebula",
                        tags: [],
                        title: "Scatter Nebula",
                        titleCN: "大规模星云散点图",
                        difficulty: 5
                    }, {
                        category: ["scatter"],
                        id: "scatter-stream-visual",
                        tags: [],
                        title: "Visual interaction with stream",
                        titleCN: "流式渲染和视觉映射操作",
                        difficulty: 5
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-drink",
                        tags: [],
                        title: "Drink Flavors",
                        titleCN: "Drink Flavors",
                        difficulty: 5
                    }, {
                        category: ["custom", "dataZoom"],
                        id: "wind-barb",
                        tags: [],
                        title: "Wind Barb",
                        titleCN: "风向图",
                        difficulty: 5
                    }, {
                        category: ["bar"],
                        id: "bar-race-country",
                        tags: [],
                        title: "Bar Race",
                        titleCN: "动态排序柱状图 - 人均收入",
                        difficulty: 6
                    }, {
                        category: ["bar", "rich"],
                        id: "bar-rich-text",
                        tags: [],
                        title: "Wheater Statistics",
                        titleCN: "天气统计（富文本）",
                        difficulty: 6
                    }, {
                        category: ["scatter"],
                        id: "bubble-gradient",
                        tags: [],
                        title: "Bubble Chart",
                        titleCN: "气泡图",
                        difficulty: 6
                    }, {
                        category: ["calendar", "pie"],
                        id: "calendar-pie",
                        tags: [],
                        title: "Calendar Pie",
                        titleCN: "日历饼图",
                        difficulty: 6
                    }, {
                        category: ["custom", "map"],
                        id: "custom-hexbin",
                        tags: [],
                        title: "Hexagonal Binning",
                        titleCN: "六边形分箱图（自定义系列）",
                        difficulty: 6
                    }, {
                        category: ["bar"],
                        id: "dynamic-data",
                        tags: [],
                        title: "Dynamic Data",
                        titleCN: "动态数据",
                        difficulty: 6
                    }, {
                        category: ["gauge"],
                        id: "gauge-barometer",
                        tags: [],
                        title: "Gauge Barometer chart",
                        titleCN: "气压表",
                        difficulty: 6
                    }, {
                        category: ["graph"],
                        id: "graph-force-dynamic",
                        tags: [],
                        title: "Graph Dynamic",
                        titleCN: "动态增加图节点",
                        difficulty: 6
                    }, {
                        category: ["line"],
                        id: "line-markline",
                        tags: [],
                        title: "Line with Marklines",
                        titleCN: "折线图的标记线",
                        difficulty: 6
                    }, {
                        category: ["line"],
                        id: "line-style",
                        tags: [],
                        title: "Line Style and Item Style",
                        titleCN: "自定义折线图样式",
                        difficulty: 6
                    }, {
                        category: ["bar"],
                        id: "mix-timeline-finance",
                        tags: [],
                        title: "Finance Indices 2002",
                        titleCN: "2002全国宏观经济指标",
                        difficulty: 6
                    }, {
                        category: ["sunburst"],
                        id: "sunburst-book",
                        tags: [],
                        title: "Book Records",
                        titleCN: "书籍分布",
                        difficulty: 6
                    }, {
                        category: ["bar"],
                        id: "watermark",
                        tags: [],
                        title: "Watermark - ECharts Download",
                        titleCN: "水印 - ECharts 下载统计",
                        difficulty: 6
                    }, {
                        category: ["bar"],
                        id: "bar-polar-real-estate",
                        tags: [],
                        title: "Bar Chart on Polar",
                        difficulty: 7
                    }, {
                        category: ["bar"],
                        id: "bar-polar-stack",
                        tags: [],
                        title: "Stacked Bar Chart on Polar",
                        titleCN: "极坐标系下的堆叠柱状图",
                        difficulty: 7
                    }, {
                        category: ["bar"],
                        id: "bar-polar-stack-radial",
                        tags: [],
                        title: "Stacked Bar Chart on Polar(Radial)",
                        titleCN: "极坐标系下的堆叠柱状图",
                        difficulty: 7
                    }, {
                        category: ["custom", "calendar"],
                        id: "custom-calendar-icon",
                        tags: [],
                        title: "Custom Calendar Icon",
                        titleCN: "日历图自定义图标",
                        difficulty: 7
                    }, {
                        category: ["custom"],
                        id: "custom-wind",
                        tags: [],
                        title: "Use custom series to draw wind vectors",
                        titleCN: "使用自定义系列绘制风场",
                        difficulty: 7
                    }, {
                        category: ["gauge"],
                        id: "gauge-clock",
                        tags: [],
                        title: "Clock Gauge",
                        titleCN: "时钟仪表盘",
                        difficulty: 7
                    }, {
                        category: ["graph"],
                        id: "graph-life-expectancy",
                        tags: [],
                        title: "Graph Life Expectancy",
                        titleCN: "Graph Life Expectancy",
                        difficulty: 7
                    }, {
                        category: ["line"],
                        id: "line-in-cartesian-coordinate-system",
                        tags: [],
                        title: "Line Chart in Cartesian Coordinate System",
                        titleCN: "双数值轴折线图",
                        difficulty: 7
                    }, {
                        category: ["line"],
                        id: "line-log",
                        tags: [],
                        title: "Log Axis",
                        titleCN: "对数轴示例",
                        difficulty: 7
                    }, {
                        category: ["line"],
                        id: "line-step",
                        tags: [],
                        title: "Step Line",
                        titleCN: "阶梯折线图",
                        difficulty: 7
                    }, {
                        category: ["bar"],
                        id: "polar-roundCap",
                        tags: [],
                        title: "Rounded Bar on Polar",
                        titleCN: "圆角环形图",
                        difficulty: 7
                    }, {
                        category: ["scatter"],
                        id: "scatter-aqi-color",
                        tags: [],
                        title: "Scatter Aqi Color",
                        titleCN: "AQI 气泡图",
                        difficulty: 7
                    }, {
                        category: ["scatter"],
                        id: "scatter-nutrients",
                        tags: [],
                        title: "Scatter Nutrients",
                        titleCN: "营养分布散点图",
                        difficulty: 7
                    }, {
                        category: ["scatter"],
                        id: "scatter-nutrients-matrix",
                        tags: [],
                        title: "Scatter Nutrients Matrix",
                        titleCN: "营养分布散点矩阵",
                        difficulty: 7
                    }, {
                        category: ["gauge"],
                        id: "gauge-car",
                        tags: [],
                        title: "Gauge Car",
                        titleCN: "Gauge Car",
                        difficulty: 8
                    }, {
                        category: ["graph"],
                        id: "graph-webkit-dep",
                        tags: [],
                        title: "Graph Webkit Dep",
                        titleCN: "WebKit 模块关系依赖图",
                        difficulty: 8
                    }, {
                        category: ["line"],
                        id: "line-easing",
                        tags: [],
                        title: "Line Easing Visualizing",
                        titleCN: "缓动函数可视化",
                        difficulty: 8
                    }, {
                        category: ["line"],
                        id: "line-y-category",
                        tags: [],
                        title: "Line Y Category",
                        titleCN: "垂直折线图（Y轴为类目轴）",
                        difficulty: 8
                    }, {
                        category: ["scatter"],
                        id: "scatter-polar-punchCard",
                        tags: [],
                        title: "Punch Card of Github",
                        titleCN: "GitHub 打卡气泡图（极坐标）",
                        difficulty: 8
                    }, {
                        category: ["custom"],
                        id: "custom-aggregate-scatter-bar",
                        tags: [],
                        title: "Aggregate Morphing Between Scatter and Bar",
                        titleCN: "聚合分割形变（散点图 - 柱状图）",
                        difficulty: 9
                    }, {
                        category: ["custom"],
                        id: "custom-aggregate-scatter-pie",
                        tags: [],
                        title: "Aggregate Morphing Between Scatter and Pie",
                        titleCN: "聚合分割形变（散点图 - 饼图）",
                        difficulty: 9
                    }, {
                        category: ["custom"],
                        id: "custom-gauge",
                        tags: [],
                        title: "Custom Gauge",
                        titleCN: "自定义仪表",
                        difficulty: 9
                    }, {
                        category: ["graph"],
                        id: "graph-npm",
                        tags: [],
                        title: "NPM Dependencies",
                        titleCN: "NPM 依赖关系图",
                        difficulty: 9
                    }, {
                        category: ["line"],
                        id: "line-graphic",
                        tags: [],
                        title: "Custom Graphic Component",
                        titleCN: "自定义图形组件",
                        difficulty: 9
                    }, {
                        category: ["line"],
                        id: "line-pen",
                        tags: [],
                        title: "Click to Add Points",
                        titleCN: "点击添加折线图拐点",
                        difficulty: 9
                    }, {
                        category: ["scatter"],
                        id: "scatter-life-expectancy-timeline",
                        tags: [],
                        title: "Life Expectancy and GDP",
                        titleCN: "各国人均寿命与GDP关系演变",
                        difficulty: 9
                    }, {
                        category: ["scatter"],
                        id: "scatter-painter-choice",
                        tags: [],
                        title: "Master Painter Color Choices Throughout History",
                        titleCN: "Master Painter Color Choices Throughout History",
                        difficulty: 9
                    }, {
                        category: ["boxplot"],
                        id: "boxplot-light-velocity",
                        tags: [],
                        title: "Boxplot Light Velocity",
                        titleCN: "基础盒须图",
                        difficulty: 10
                    }, {
                        category: ["boxplot"],
                        id: "boxplot-light-velocity2",
                        tags: [],
                        title: "Boxplot Light Velocity2",
                        titleCN: "垂直方向盒须图",
                        difficulty: 10
                    }, {
                        category: ["boxplot"],
                        id: "boxplot-multi",
                        tags: [],
                        title: "Multiple Categories",
                        titleCN: "多系列盒须图",
                        difficulty: 10
                    }, {
                        category: [],
                        id: "calendar-effectscatter",
                        tags: [],
                        difficulty: 10
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-brush",
                        tags: [],
                        title: "Candlestick Brush",
                        titleCN: "日力图刷选",
                        difficulty: 10
                    }, {
                        category: ["candlestick"],
                        id: "candlestick-sh-2015",
                        tags: [],
                        title: "ShangHai Index, 2015",
                        titleCN: "2015 年上证指数",
                        difficulty: 10
                    }, {
                        category: [],
                        id: "covid-america",
                        tags: [],
                        difficulty: 10
                    }, {
                        category: ["custom"],
                        id: "custom-aggregate-scatter-cluster",
                        tags: [],
                        title: "Aggregate Morphing Between Scatter Clustering",
                        titleCN: "聚合分割形变（散点图聚类）",
                        difficulty: 10
                    }, {
                        category: ["scatter", "map"],
                        id: "effectScatter-bmap",
                        tags: ["bmap"],
                        title: "Air Quality - Baidu Map",
                        titleCN: "全国主要城市空气质量 - 百度地图",
                        difficulty: 10
                    }, {
                        category: ["funnel"],
                        id: "funnel",
                        tags: [],
                        title: "Funnel Chart",
                        titleCN: "漏斗图",
                        difficulty: 10
                    }, {
                        category: ["funnel"],
                        id: "funnel-align",
                        tags: [],
                        title: "Funnel (align)",
                        titleCN: "漏斗图(对比)",
                        difficulty: 10
                    }, {
                        category: ["funnel"],
                        id: "funnel-customize",
                        tags: [],
                        title: "Customized Funnel",
                        titleCN: "漏斗图",
                        difficulty: 10
                    }, {
                        category: ["funnel"],
                        id: "funnel-mutiple",
                        tags: [],
                        title: "Multiple Funnels",
                        titleCN: "漏斗图",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-beef-cuts",
                        tags: [],
                        title: "GEO Beef Cuts",
                        titleCN: "庖丁解牛",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-lines",
                        tags: [],
                        title: "Migration",
                        titleCN: "模拟迁徙",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-map-scatter",
                        tags: [],
                        title: "map and scatter share a geo",
                        titleCN: "map and scatter share a geo",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-organ",
                        tags: [],
                        title: "Organ Data with SVG",
                        titleCN: "内脏数据（SVG）",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-seatmap-flight",
                        tags: [],
                        title: "Flight Seatmap with SVG",
                        titleCN: "航班选座（SVG）",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-svg-lines",
                        tags: [],
                        title: "GEO SVG Lines",
                        titleCN: "GEO 路径图（SVG）",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-svg-map",
                        tags: [],
                        title: "GEO SVG Map",
                        titleCN: "地图（SVG）",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "geo-svg-traffic",
                        tags: [],
                        title: "GEO SVG Traffic",
                        titleCN: "交通（SVG）",
                        difficulty: 10
                    }, {
                        category: ["line", "drag"],
                        id: "line-draggable",
                        tags: [],
                        title: "Try Dragging these Points",
                        titleCN: "可拖拽点",
                        difficulty: 10
                    }, {
                        category: ["line"],
                        id: "line-polar",
                        tags: [],
                        title: "Two Value-Axes in Polar",
                        titleCN: "极坐标双数值轴",
                        difficulty: 10
                    }, {
                        category: ["line"],
                        id: "line-polar2",
                        tags: [],
                        title: "Two Value-Axes in Polar",
                        titleCN: "极坐标双数值轴",
                        difficulty: 10
                    }, {
                        category: ["line", "dataZoom"],
                        id: "line-tooltip-touch",
                        tags: [],
                        title: "Tooltip and DataZoom on Mobile",
                        titleCN: "移动端上的 dataZoom 和 tooltip",
                        difficulty: 10
                    }, {
                        category: ["map", "lines"],
                        id: "lines-airline",
                        tags: [],
                        title: "65k+ Airline",
                        titleCN: "65k+ 飞机航线",
                        difficulty: 10
                    }, {
                        category: ["map", "lines"],
                        id: "lines-bmap",
                        tags: ["bmap"],
                        title: "A Hiking Trail in Hangzhou - Baidu Map",
                        titleCN: "杭州热门步行路线 - 百度地图",
                        difficulty: 10
                    }, {
                        category: ["map", "lines"],
                        id: "lines-bmap-bus",
                        tags: ["bmap"],
                        title: "Bus Lines of Beijing - Baidu Map",
                        titleCN: "北京公交路线 - 百度地图",
                        difficulty: 10
                    }, {
                        category: ["map", "lines"],
                        id: "lines-bmap-effect",
                        tags: ["bmap"],
                        title: "Bus Lines of Beijing - Line Effect",
                        titleCN: "北京公交路线 - 线特效",
                        difficulty: 10
                    }, {
                        category: ["map", "lines"],
                        id: "lines-ny",
                        tags: [],
                        title: "Use lines to draw 1 million ny streets.",
                        titleCN: "使用线图绘制近 100 万的纽约街道数据",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-bin",
                        tags: ["bmap"],
                        title: "Binning on Map",
                        titleCN: "Binning on Map",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-china",
                        tags: [],
                        title: "Map China",
                        titleCN: "Map China",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-china-dataRange",
                        tags: [],
                        title: "Sales of iphone",
                        titleCN: "iphone销量",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-HK",
                        tags: [],
                        title: "Population Density of HongKong (2011)",
                        titleCN: "香港18区人口密度 （2011）",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-labels",
                        tags: [],
                        title: "Rich Text Labels on Map",
                        titleCN: "地图上的富文本标签",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-locate",
                        tags: [],
                        title: "Map Locate",
                        titleCN: "Map Locate",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-polygon",
                        tags: ["bmap"],
                        title: "Draw Polygon on Map",
                        titleCN: "在地图上绘制多边形",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-province",
                        tags: [],
                        title: "Switch among 34 Provinces",
                        titleCN: "34 省切换查看",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-usa",
                        tags: [],
                        title: "USA Population Estimates (2012)",
                        titleCN: "USA Population Estimates (2012)",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-world",
                        tags: [],
                        title: "Map World",
                        titleCN: "Map World",
                        difficulty: 10
                    }, {
                        category: ["map"],
                        id: "map-world-dataRange",
                        tags: [],
                        title: "World Population (2010)",
                        titleCN: "World Population (2010)",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-body-fill",
                        tags: [],
                        title: "Water Content",
                        titleCN: "人体含水量",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-dotted",
                        tags: [],
                        title: "Dotted bar",
                        titleCN: "虚线柱状图效果",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-forest",
                        tags: [],
                        title: "Expansion of forest",
                        titleCN: "森林的增长",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-hill",
                        tags: [],
                        title: "Wish List and Mountain Height",
                        titleCN: "圣诞愿望清单和山峰高度",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-spirit",
                        tags: [],
                        title: "Spirits",
                        titleCN: "精灵",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-vehicle",
                        tags: [],
                        title: "Vehicles",
                        titleCN: "交通工具",
                        difficulty: 10
                    }, {
                        category: ["pictorialBar"],
                        id: "pictorialBar-velocity",
                        tags: [],
                        title: "Velocity of Christmas Reindeers",
                        titleCN: "驯鹿的速度",
                        difficulty: 10
                    }, {
                        category: ["radar"],
                        id: "radar-multiple",
                        tags: [],
                        title: "Multiple Radar",
                        titleCN: "多雷达图",
                        difficulty: 10
                    }, {
                        category: ["scatter"],
                        id: "scatter-map",
                        tags: [],
                        title: "Air Quality",
                        titleCN: "全国主要城市空气质量",
                        difficulty: 10
                    }, {
                        category: ["scatter"],
                        id: "scatter-map-brush",
                        tags: [],
                        title: "Scatter Map Brush",
                        titleCN: "Scatter Map Brush",
                        difficulty: 10
                    }, {
                        category: ["parallel", "scatter"],
                        id: "scatter-matrix",
                        tags: [],
                        title: "Scatter Matrix",
                        titleCN: "散点矩阵和平行坐标",
                        difficulty: 10
                    }, {
                        category: ["scatter"],
                        id: "scatter-weibo",
                        tags: [],
                        title: "Sign in of weibo",
                        titleCN: "微博签到数据点亮中国",
                        difficulty: 10
                    }, {
                        category: ["scatter"],
                        id: "scatter-world-population",
                        tags: [],
                        title: "World Population (2011)",
                        titleCN: "World Population (2011)",
                        difficulty: 10
                    }, {
                        category: ["themeRiver"],
                        id: "themeRiver-basic",
                        tags: [],
                        title: "ThemeRiver",
                        titleCN: "主题河流图",
                        difficulty: 10
                    }, {
                        category: ["themeRiver"],
                        id: "themeRiver-lastfm",
                        tags: [],
                        title: "ThemeRiver Lastfm",
                        titleCN: "ThemeRiver Lastfm",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-basic",
                        tags: [],
                        title: "From Left to Right Tree",
                        titleCN: "从左到右树状图",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-legend",
                        tags: [],
                        title: "Multiple Trees",
                        titleCN: "多棵树",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-orient-bottom-top",
                        tags: [],
                        title: "From Bottom to Top Tree",
                        titleCN: "从下到上树状图",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-orient-right-left",
                        tags: [],
                        title: "From Right to Left Tree",
                        titleCN: "从右到左树状图",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-polyline",
                        tags: [],
                        title: "Tree with Polyline Edge",
                        titleCN: "折线树图",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-radial",
                        tags: [],
                        title: "Radial Tree",
                        titleCN: "径向树状图",
                        difficulty: 10
                    }, {
                        category: ["tree"],
                        id: "tree-vertical",
                        tags: [],
                        title: "From Top to Bottom Tree",
                        titleCN: "从上到下树状图",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-disk",
                        tags: [],
                        title: "Disk Usage",
                        titleCN: "磁盘占用",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-drill-down",
                        tags: [],
                        title: "ECharts Option Query",
                        titleCN: "ECharts 配置项查询分布",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-obama",
                        tags: [],
                        title: "How $3.7 Trillion is Spent",
                        titleCN: "How $3.7 Trillion is Spent",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-show-parent",
                        tags: [],
                        title: "Show Parent Labels",
                        titleCN: "显示父层级标签",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-simple",
                        tags: [],
                        title: "Basic Treemap",
                        titleCN: "基础矩形树图",
                        difficulty: 10
                    }, {
                        category: ["treemap"],
                        id: "treemap-visual",
                        tags: [],
                        title: "Gradient Mapping",
                        titleCN: "映射为渐变色",
                        difficulty: 10
                    }, {
                        category: ["calendar", "scatter"],
                        id: "calendar-charts",
                        tags: [],
                        title: "Calendar Charts",
                        titleCN: "日力图",
                        difficulty: 11
                    }, {
                        category: ["custom"],
                        id: "circle-packing-with-d3",
                        tags: [],
                        title: "Circle Packing with d3",
                        titleCN: "Circle Packing with d3",
                        difficulty: 11
                    }, {
                        category: ["custom"],
                        id: "custom-one-to-one-morph",
                        tags: [],
                        title: "One-to-one Morphing",
                        titleCN: "一对一映射形变",
                        difficulty: 11
                    }, {
                        category: ["custom"],
                        id: "custom-spiral-race",
                        tags: [],
                        title: "Custom Spiral Race",
                        titleCN: "自定义螺旋线竞速",
                        difficulty: 11
                    }, {
                        category: ["custom"],
                        id: "custom-story-transition",
                        tags: [],
                        title: "Simple Story Transition",
                        titleCN: "极简场景变换示例",
                        difficulty: 11
                    }, {
                        category: ["scatter"],
                        id: "scatter-logarithmic-regression",
                        tags: [],
                        title: "Logarithmic Regression",
                        titleCN: "对数回归（使用统计插件）",
                        difficulty: 16
                    }],
                    Rt = [{
                        category: ["globe"],
                        id: "animating-contour-on-globe",
                        tags: [],
                        title: "Animating Contour on Globe",
                        titleCN: "Animating Contour on Globe",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-dataset",
                        tags: [],
                        title: "3D Bar with Dataset",
                        titleCN: "使用 dataset 为三维柱状图设置数据",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-global-population",
                        tags: [],
                        title: "Bar3D - Global Population",
                        titleCN: "Bar3D - Global Population",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-myth",
                        tags: [],
                        title: "星云",
                        titleCN: "星云",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-noise-modified-from-marpi-demo",
                        tags: [],
                        title: "Noise modified from marpi's demo",
                        titleCN: "Noise modified from marpi's demo",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-punch-card",
                        tags: [],
                        title: "Bar3D - Punch Card",
                        titleCN: "Bar3D - Punch Card",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-simplex-noise",
                        tags: [],
                        theme: "dark",
                        title: "Bar3D - Simplex Noise",
                        titleCN: "Bar3D - Simplex Noise",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "bar3d-voxelize-image",
                        tags: [],
                        title: "Voxelize image",
                        titleCN: "Voxelize image",
                        difficulty: 10
                    }, {
                        category: ["flowGL"],
                        id: "flowGL-noise",
                        tags: [],
                        theme: "dark",
                        title: "Flow on the cartesian",
                        titleCN: "直角坐标系上的向量场",
                        difficulty: 10
                    }, {
                        category: ["geo3D"],
                        id: "geo3d",
                        tags: [],
                        title: "Geo3D",
                        titleCN: "Geo3D",
                        difficulty: 10
                    }, {
                        category: ["geo3D"],
                        id: "geo3d-with-different-height",
                        tags: [],
                        title: "Geo3D with Different Height",
                        titleCN: "Geo3D with Different Height",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "global-population-bar3d-on-globe",
                        tags: [],
                        title: "Global Population - Bar3D on Globe",
                        titleCN: "Global Population - Bar3D on Globe",
                        difficulty: 10
                    }, {
                        category: ["flowGL"],
                        id: "global-wind-visualization",
                        tags: ["bmap"],
                        title: "Global wind visualization",
                        titleCN: "Global wind visualization",
                        difficulty: 10
                    }, {
                        category: ["flowGL"],
                        id: "global-wind-visualization-2",
                        tags: ["bmap"],
                        title: "Global Wind Visualization 2",
                        titleCN: "Global Wind Visualization 2",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-atmosphere",
                        tags: [],
                        title: "Globe with Atmosphere",
                        titleCN: "大气层显示",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-contour-paint",
                        tags: [],
                        title: "Contour Paint",
                        titleCN: "Contour Paint",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-country-carousel",
                        tags: [],
                        title: "Country Carousel",
                        titleCN: "Country Carousel",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-displacement",
                        tags: [],
                        title: "Globe Displacement",
                        titleCN: "Globe Displacement",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-echarts-gl-hello-world",
                        tags: [],
                        title: "ECharts-GL Hello World",
                        titleCN: "ECharts-GL Hello World",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-layers",
                        tags: [],
                        title: "Globe Layers",
                        titleCN: "Globe Layers",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-moon",
                        tags: [],
                        title: "Moon",
                        titleCN: "Moon",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "globe-with-echarts-surface",
                        tags: [],
                        title: "Globe with ECharts Surface",
                        titleCN: "Globe with ECharts Surface",
                        difficulty: 10
                    }, {
                        category: ["graphGL"],
                        id: "graphgl-gpu-layout",
                        tags: [],
                        theme: "dark",
                        title: "GraphGL GPU Layout",
                        titleCN: "GraphGL GPU Layout",
                        difficulty: 10
                    }, {
                        category: ["graphGL"],
                        id: "graphgl-large-internet",
                        tags: [],
                        theme: "dark",
                        title: "GraphGL - Large Internet",
                        titleCN: "GraphGL - Large Internet",
                        difficulty: 10
                    }, {
                        category: ["graphGL"],
                        id: "graphgl-npm-dep",
                        tags: [],
                        theme: "dark",
                        title: "NPM Dependencies with graphGL",
                        titleCN: "1w 节点 2w7 边的NPM 依赖图",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "image-surface-sushuang",
                        tags: [],
                        title: "Image Surface Sushuang",
                        titleCN: "Image Surface Sushuang",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "image-to-bar3d",
                        tags: [],
                        title: "Image to Bar3D",
                        titleCN: "Image to Bar3D",
                        difficulty: 10
                    }, {
                        category: ["globe"],
                        id: "iron-globe",
                        tags: [],
                        title: "Iron globe",
                        titleCN: "Iron globe",
                        difficulty: 10
                    }, {
                        category: ["line3D"],
                        id: "line3d-orthographic",
                        tags: [],
                        title: "三维折线图正交投影",
                        titleCN: "三维折线图正交投影",
                        difficulty: 10
                    }, {
                        category: ["lines3D"],
                        id: "lines3d-airline-on-globe",
                        tags: [],
                        title: "Airline on Globe",
                        titleCN: "Airline on Globe",
                        difficulty: 10
                    }, {
                        category: ["lines3D"],
                        id: "lines3d-flights",
                        tags: [],
                        title: "Flights",
                        titleCN: "Flights",
                        difficulty: 10
                    }, {
                        category: ["lines3D"],
                        id: "lines3d-flights-gl",
                        tags: [],
                        title: "Flights GL",
                        titleCN: "Flights GL",
                        difficulty: 10
                    }, {
                        category: ["lines3D"],
                        id: "lines3d-flights-on-geo3d",
                        tags: [],
                        title: "Flights on Geo3D",
                        titleCN: "Flights on Geo3D",
                        difficulty: 10
                    }, {
                        category: ["linesGL"],
                        id: "linesGL-ny",
                        tags: [],
                        title: "Use linesGL to draw 1 million ny streets.",
                        titleCN: "实时交互的纽约街道可视化",
                        difficulty: 10
                    }, {
                        category: ["map3D"],
                        id: "map3d-alcohol-consumption",
                        tags: [],
                        title: "Map3D - Alcohol Consumption",
                        titleCN: "Map3D - Alcohol Consumption",
                        difficulty: 10
                    }, {
                        category: ["map3D"],
                        id: "map3d-buildings",
                        tags: [],
                        title: "Buildings",
                        titleCN: "Buildings",
                        difficulty: 10
                    }, {
                        category: ["map3D"],
                        id: "map3d-wood-city",
                        tags: [],
                        title: "Wood City",
                        titleCN: "Wood City",
                        difficulty: 10
                    }, {
                        category: ["map3D"],
                        id: "map3d-wood-map",
                        tags: [],
                        title: "木质世界地图",
                        titleCN: "木质世界地图",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "metal-bar3d",
                        tags: [],
                        title: "Metal Bar3D",
                        titleCN: "Metal Bar3D",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "metal-surface",
                        tags: [],
                        title: "Metal Surface",
                        titleCN: "Metal Surface",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "parametric-surface-rose",
                        tags: [],
                        title: "Parametric Surface Rose",
                        titleCN: "Parametric Surface Rose",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3d",
                        tags: [],
                        theme: "dark",
                        title: "Scatter3D",
                        titleCN: "Scatter3D",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3D-dataset",
                        tags: [],
                        title: "3D Scatter with Dataset",
                        titleCN: "使用 dataset 为三维散点图设置数据",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3d-globe-population",
                        tags: [],
                        title: "Scatter3D - Globe Population",
                        titleCN: "Scatter3D - Globe Population",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3d-orthographic",
                        tags: [],
                        theme: "dark",
                        title: "三维散点图正交投影",
                        titleCN: "三维散点图正交投影",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3d-scatter",
                        tags: [],
                        title: "3D Scatter with Scatter Matrix",
                        titleCN: "三维散点图和散点矩阵结合使用",
                        difficulty: 10
                    }, {
                        category: ["scatter3D"],
                        id: "scatter3d-simplex-noise",
                        tags: [],
                        theme: "dark",
                        title: "Scatter3D - Simplex Noise",
                        titleCN: "Scatter3D - Simplex Noise",
                        difficulty: 10
                    }, {
                        category: ["scatterGL"],
                        id: "scatterGL-gps",
                        tags: [],
                        title: "10 million Bulk GPS points",
                        titleCN: "1 千万 GPS 点可视化",
                        difficulty: 10
                    }, {
                        category: ["scatterGL"],
                        id: "scattergl-weibo",
                        tags: [],
                        theme: "dark",
                        title: "微博签到数据点亮中国",
                        titleCN: "微博签到数据点亮中国",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "simple-surface",
                        tags: [],
                        title: "Simple Surface",
                        titleCN: "Simple Surface",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "sphere-parametric-surface",
                        tags: [],
                        title: "Sphere Parametric Surface",
                        titleCN: "Sphere Parametric Surface",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "stacked-bar3d",
                        tags: [],
                        title: "Stacked Bar3D",
                        titleCN: "Stacked Bar3D",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-breather",
                        tags: [],
                        title: "Breather",
                        titleCN: "Breather",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-golden-rose",
                        tags: [],
                        title: "Golden Rose",
                        titleCN: "Golden Rose",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-leather",
                        tags: [],
                        title: "皮革材质",
                        titleCN: "皮革材质",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-mollusc-shell",
                        tags: [],
                        title: "Mollusc Shell",
                        titleCN: "Mollusc Shell",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-theme-roses",
                        tags: [],
                        title: "Theme Roses",
                        titleCN: "Theme Roses",
                        difficulty: 10
                    }, {
                        category: ["surface"],
                        id: "surface-wave",
                        tags: [],
                        title: "Surface Wave",
                        titleCN: "Surface Wave",
                        difficulty: 10
                    }, {
                        category: ["bar3D"],
                        id: "transparent-bar3d",
                        tags: [],
                        title: "Transparent Bar3D",
                        titleCN: "Transparent Bar3D",
                        difficulty: 10
                    }];
                var It = ut.sourceCode.indexOf("ROOT_PATH") >= 0 ? "var ROOT_PATH = '".concat(ut.cdnRoot, "'") : "";

                function Bt(t) {
                    return function(t) {
                        if (Array.isArray(t)) return jt(t)
                    }(t) || function(t) {
                        if ("undefined" != typeof Symbol && Symbol.iterator in Object(t)) return Array.from(t)
                    }(t) || function(t, e) {
                        if (t) {
                            if ("string" == typeof t) return jt(t, e);
                            var n = Object.prototype.toString.call(t).slice(8, -1);
                            return "Object" === n && t.constructor && (n = t.constructor.name), "Map" === n || "Set" === n ? Array.from(t) : "Arguments" === n || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? jt(t, e) : void 0
                        }
                    }(t) || function() {
                        throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")
                    }()
                }

                function jt(t, e) {
                    (null == e || e > t.length) && (e = t.length);
                    for (var n = 0, a = new Array(e); n < e; n++) a[n] = t[n];
                    return a
                }

                function Gt(t) {
                    return rt.c === t.id
                }
                var zt = Ft.concat(Rt).find(Gt),
                    $t = Rt.find(Gt);

                function Ut(t) {
                    ut.enableDecal && (t.aria = t.aria || {}, t.aria.decal = t.aria.decal || {}, t.aria.decal.show = !0, t.aria.show = t.aria.enabled = !0)
                }

                function Zt() {
                    if ("undefined" == typeof echarts) {
                        var t = zt && zt.tags.indexOf("bmap") >= 0;
                        return t && (window.HOST_TYPE = "2", window.BMap_loadScriptTime = (new Date).getTime()), at([ct.datGUIMinJS, "local" in rt ? ct.localEChartsMinJS : ct.echartsMinJS, ct.echartsDir + "/dist/extension/dataTool.js", "https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/js/world.js", ct.echartsStatMinJS].concat(Bt(rt.gl ? [ct.echartsGLMinJS] : []), Bt(t ? ["https://api.map.baidu.com/getscript?v=3.0&ak=KOmVjPVUAey1G2E8zNhPiuQ6QiEmAwZu&services=&t=20200327103013", ct.echartsDir + "/dist/extension/bmap.js"] : []))).then((function() {
                            echarts.registerPreprocessor(Ut)
                        }))
                    }
                    return Promise.resolve()
                }

                function Vt(t, e) {
                    "warn" !== e && "error" !== e && (e = "info"), ut.editorStatus.message = t, ut.editorStatus.type = e
                }
                var Wt = gt({
                    props: ["inEditor"],
                    data: function() {
                        return {
                            shared: ut,
                            debouncedTime: void 0,
                            backgroundColor: "",
                            autoRun: !0,
                            loading: !1,
                            isGL: $t
                        }
                    },
                    mounted: function() {
                        var t = this;
                        this.loading = !0, Zt().then((function() {
                            t.loading = !1, ut.runCode && t.run()
                        })), Ot(this.$el, (function() {
                            t.sandbox && t.sandbox.resize()
                        }))
                    },
                    computed: {
                        editLink: function() {
                            var t = ["c=" + rt.c];
                            return rt.theme && t.push(["theme=" + rt.theme]), rt.gl && t.push(["gl=" + rt.gl]), "./editor.html?" + t.join("&")
                        }
                    },
                    watch: {
                        "shared.runCode": function(t) {
                            !this.autoRun && this.sandbox || (this.debouncedRun ? this.debouncedRun() : this.run())
                        },
                        "shared.renderer": function() {
                            this.refreshAll()
                        },
                        "shared.darkMode": function() {
                            this.refreshAll()
                        },
                        "shared.enableDecal": function() {
                            this.refreshAll()
                        },
                        "shared.useDirtyRect": function() {
                            this.refreshAll()
                        }
                    },
                    methods: {
                        run: function t() {
                            var e = this;
                            if ("undefined" != typeof echarts) {
                                this.sandbox || (this.sandbox = function(t) {
                                    var e, n = {},
                                        a = [],
                                        i = [],
                                        o = window.setTimeout,
                                        r = window.setInterval;

                                    function l(t, e) {
                                        var n = o(t, e);
                                        return i.push(n), n
                                    }

                                    function s(t, e) {
                                        var n = r(t, e);
                                        return a.push(n), n
                                    }
                                    var c, u = [];
                                    return {
                                        resize: function() {
                                            c && c.resize()
                                        },
                                        dispose: function() {
                                            c && (c.dispose(), c = null)
                                        },
                                        getDataURL: function() {
                                            return c.getDataURL({
                                                pixelRatio: 2,
                                                excludeComponents: ["toolbox"]
                                            })
                                        },
                                        getOption: function() {
                                            return c.getOption()
                                        },
                                        run: function(o, r) {
                                            if (!c) {
                                                if (c = echarts.init(o, r.darkMode ? "dark" : "", {
                                                        renderer: r.renderer,
                                                        useDirtyRect: r.useDirtyRect
                                                    }), r.useDirtyRect && "canvas" === r.renderer) try {
                                                    ! function(t, e) {
                                                        e = e || {};
                                                        var n = t.painter;
                                                        if (!n.getLayers) throw new Error("Debug dirty rect can only been used on canvas renderer.");
                                                        if (n.isSingleCanvas()) throw new Error("Debug dirty rect can only been used on zrender inited with container.");
                                                        var a = document.createElement("div");
                                                        a.style.cssText = "\nposition:absolute;\nleft:0;\ntop:0;\nright:0;\nbottom:0;\npointer-events:none;\n", a.className = "ec-debug-dirty-rect-container";
                                                        var i = [],
                                                            o = t.dom;
                                                        o.appendChild(a), "static" === getComputedStyle(o).position && (o.style.position = "relative"), t.on("rendered", (function() {
                                                            if (n.getLayers) {
                                                                var t = 0;
                                                                n.eachBuiltinLayer((function(n) {
                                                                    if (n.debugGetPaintRects)
                                                                        for (var o = n.debugGetPaintRects(), r = 0; r < o.length; r++) i[t] || (i[t] = new vt(e.style), a.appendChild(i[t].dom)), i[t].show(), i[t].update(o[r]), t++
                                                                }));
                                                                for (var o = t; o < i.length; o++) i[o].hide()
                                                            }
                                                        }))
                                                    }(c.getZr(), {
                                                        autoHideDelay: 500
                                                    })
                                                } catch (t) {
                                                    console.error(t)
                                                }
                                                p = (d = c).on, f = d.setOption, d.on = function(t) {
                                                    var e = p.apply(d, arguments);
                                                    return u.push(t), e
                                                }, d.setOption = function() {
                                                    var e = f.apply(this, arguments);
                                                    return t && t(d), e
                                                }
                                            }
                                            var d, p, f;
                                            ! function() {
                                                for (var t = 0; t < a.length; t++) clearInterval(a[t]);
                                                for (t = 0; t < i.length; t++) clearTimeout(i[t]);
                                                a = [], i = []
                                            }(),
                                            function(t) {
                                                u.forEach((function(e) {
                                                    t && t.off(e)
                                                })), u.length = 0
                                            }(c), n.config = null;
                                            var g = r.runCode,
                                                m = new Function("myChart", "app", "setTimeout", "setInterval", "ROOT_PATH", "var option;\n" + g + "\nreturn option;")(c, n, l, s, r.cdnRoot),
                                                h = 0;
                                            if (m && "object" === bt(m)) {
                                                var C = +new Date;
                                                c.setOption(m, !0), h = +new Date - C
                                            }
                                            if (e && ($(e.domElement).remove(), e.destroy(), e = null), n.config) {
                                                e = new dat.GUI({
                                                    autoPlace: !1
                                                }), $(e.domElement).css({
                                                    position: "absolute",
                                                    right: 5,
                                                    top: 0,
                                                    zIndex: 1e3
                                                }), $(".right-container").append(e.domElement);
                                                var y = n.configParameters || {};
                                                for (var v in n.config) {
                                                    var b = n.config[v];
                                                    if ("onChange" !== v && "onFinishChange" !== v) {
                                                        var _ = !1,
                                                            L = null;
                                                        if (y[v] && (y[v].options ? L = e.add(n.config, v, y[v].options) : null != y[v].min && (L = e.add(n.config, v, y[v].min, y[v].max))), "string" == typeof obj) try {
                                                            var w = echarts.color.parse(b);
                                                            (_ = !!w) && (b = echarts.color.stringify(w, "rgba"))
                                                        } catch (t) {}
                                                        L || (L = e[_ ? "addColor" : "add"](n.config, v)), n.config.onChange && L.onChange(n.config.onChange), n.config.onFinishChange && L.onFinishChange(n.config.onFinishChange)
                                                    }
                                                }
                                            }
                                            return h
                                        }
                                    }
                                }((function(t) {
                                    var n = t.getOption();
                                    "string" == typeof n.backgroundColor && "transparent" !== n.backgroundColor ? e.backgroundColor = n.backgroundColor : e.backgroundColor = "#fff"
                                })));
                                try {
                                    var n = this.sandbox.run(this.$el.querySelector(".chart-container"), ut);
                                    Vt(this.$t("editor.chartOK") + n + "ms");
                                    for (var a = [0, 500, 2e3, 5e3, 1e4], i = a.length - 1; i >= 0; i--) {
                                        var o = a[i + 1] || 1e6;
                                        if (n >= a[i] && this.debouncedTime !== o) {
                                            this.debouncedRun = Lt()(t, o, {
                                                trailing: !0
                                            }), this.debouncedTime = o;
                                            break
                                        }
                                    }
                                    ut.runHash = ft++
                                } catch (t) {
                                    Vt(this.$t("editor.errorInEditor"), "error"), console.error(t)
                                }
                            }
                        },
                        refreshAll: function() {
                            this.dispose(), this.run()
                        },
                        dispose: function() {
                            this.sandbox && this.sandbox.dispose()
                        },
                        downloadExample: function() {
                            var t;
                            t = "\x3c!--\n    THIS EXAMPLE WAS DOWNLOADED FROM ".concat(window.location.href, '\n--\x3e\n<!DOCTYPE html>\n<html style="height: 100%">\n    <head>\n        <meta charset="utf-8">\n    </head>\n    <body style="height: 100%; margin: 0">\n        <div id="container" style="height: 100%"></div>\n\n        <script type="text/javascript" src="').concat(ct.echartsMinJS, '"><\/script>\n        \x3c!-- Uncomment this line if you want to dataTool extension\n        <script type="text/javascript" src="').concat(ct.echartsDir, '/dist/extension/dataTool.min.js"><\/script>\n        --\x3e\n        \x3c!-- Uncomment this line if you want to use gl extension\n        <script type="text/javascript" src="').concat(ct.echartsGLMinJS, '"><\/script>\n        --\x3e\n        \x3c!-- Uncomment this line if you want to echarts-stat extension\n        <script type="text/javascript" src="').concat(ct.echartsStatMinJS, '"><\/script>\n        --\x3e\n        \x3c!-- Uncomment this line if you want to use map\n        <script type="text/javascript" src="').concat(ct.echartsDir, '/map/js/china.js"><\/script>\n        <script type="text/javascript" src="').concat(ct.echartsDir, '/map/js/world.js"><\/script>\n        --\x3e\n        \x3c!-- Uncomment these two lines if you want to use bmap extension\n        <script type="text/javascript" src="https://api.map.baidu.com/api?v=2.0&ak=<Your Key Here>"><\/script>\n        <script type="text/javascript" src="').concat(ct.echartsDir, '/dist/extension/bmap.min.js"><\/script>\n        --\x3e\n\n        <script type="text/javascript">\nvar dom = document.getElementById("container");\nvar myChart = echarts.init(dom);\nvar app = {};\n\nvar option;\n\n').concat(It, "\n\n").concat(ut.sourceCode, "\n\nif (option && typeof option === 'object') {\n    myChart.setOption(option);\n}\n\n        <\/script>\n    </body>\n</html>\n    "),
                                function(t, e) {
                                    if ("function" == typeof window.navigator.msSaveBlob) window.navigator.msSaveOrOpenBlob(t, e);
                                    else {
                                        var n = document.createElement("a");
                                        n.href = URL.createObjectURL(t), n.download = e, n.click(), URL.revokeObjectURL(n.href)
                                    }
                                }(new Blob([t], {
                                    type: "text/html;charset=UTF-8",
                                    encoding: "UTF-8"
                                }), rt.c + ".html")
                        },
                        screenshot: function() {
                            if (this.sandbox) {
                                var t = this.sandbox.getDataURL(),
                                    e = document.createElement("a");
                                e.download = rt.c + "." + ("svg" === ut.renderer ? "svg" : "png"), e.target = "_blank", e.href = t;
                                var n = new MouseEvent("click", {
                                    bubbles: !0,
                                    cancelable: !1
                                });
                                e.dispatchEvent(n)
                            }
                        },
                        getOption: function() {
                            return this.sandbox && this.sandbox.getOption()
                        }
                    }
                }, yt, [], !1, null, null, null);
                Wt.options.__file = "src/editor/Preview.vue";
                const Ht = Wt.exports;
                var qt = gt({
                    props: ["initialCode"],
                    data: function() {
                        return {
                            shared: ut,
                            loading: !1
                        }
                    },
                    mounted: function() {
                        var t = this;
                        this.loading = !0, Zt().then((function() {
                            return "undefined" == typeof monaco ? at([ct.monacoDir + "/loader.js", ut.cdnRoot + "/js/example-transform-ts-bundle.js"]).then((function() {
                                return window.require.config({
                                    paths: {
                                        vs: ct.monacoDir
                                    }
                                }), new Promise((function(t) {
                                    window.require(["vs/editor/editor.main"], (function() {
                                        fetch(ut.cdnRoot + "/types/echarts.d.ts", {
                                            mode: "cors"
                                        }).then((function(t) {
                                            return t.text()
                                        })).then((function(t) {
                                            monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
                                                noSemanticValidation: !1,
                                                noSyntaxValidation: !1
                                            }), monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
                                                target: monaco.languages.typescript.ScriptTarget.ES6,
                                                allowNonTsExtensions: !0,
                                                noResolve: !1
                                            }), monaco.languages.typescript.typescriptDefaults.addExtraLib(t, "file:///node_modules/@types/echarts/index.d.ts"), monaco.languages.typescript.typescriptDefaults.addExtraLib("import {init, EChartsOption} from 'echarts';\n// Declare to global namespace.\ndeclare global {\ndeclare const $: any;\ndeclare const ROOT_PATH: string;\ndeclare const app: {\n    configParameters: {\n        [key: string]: ({\n            options: { [key: string]: string\n        }) | ({\n            min?: number\n            max?: number\n        })\n    }\n    config: {\n        onChange: () => void\n        [key: string]: string | number | function\n    }\n    [key: string]: any\n};\ndeclare const myChart: ReturnType<typeof init>;\ndeclare var option: EChartsOption;\n}\n", "file:///example.d.ts")
                                        })).then((function() {
                                            t()
                                        }))
                                    }))
                                }))
                            })) : Promise.resolve()
                        })).then((function() {
                            t.loading = !1;
                            var e = monaco.editor.createModel(t.initialCode || "", "typescript", monaco.Uri.parse("file:///main.ts")),
                                n = monaco.editor.create(t.$el, {
                                    model: e,
                                    fontFamily: "'Source Code Pro', 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace",
                                    minimap: {
                                        enabled: !1
                                    },
                                    automaticLayout: !0
                                });
                            t._editor = n, t.initialCode && (ut.sourceCode = t.initialCode, ut.runCode = echartsExampleTransformTs(ut.sourceCode)), n.onDidChangeModelContent((function() {
                                ut.sourceCode = n.getValue(), ut.runCode = echartsExampleTransformTs(ut.sourceCode)
                            }))
                        }))
                    },
                    destroyed: function() {
                        this._editor && (this._editor.getModel().dispose(), this._editor.dispose())
                    },
                    methods: {
                        setInitialCode: function(t) {
                            this._editor && t && this._editor.setValue(t || "")
                        }
                    },
                    watch: {
                        initialCode: function(t) {
                            this.setInitialCode(t)
                        }
                    }
                }, Ct, [], !1, null, null, null);
                qt.options.__file = "src/editor/CodeMonaco.vue";
                const Kt = qt.exports;
                var Jt = function() {
                    var t = this,
                        e = t.$createElement;
                    return (t._self._c || e)("div", {
                        directives: [{
                            name: "loading",
                            rawName: "v-loading",
                            value: t.loading,
                            expression: "loading"
                        }],
                        staticClass: "full-code-preview"
                    })
                };
                Jt._withStripped = !0;
                var Xt = gt({
                    props: ["code"],
                    data: function() {
                        return {
                            shared: ut,
                            loading: !1
                        }
                    },
                    mounted: function() {
                        var t = this;
                        this.loading = !0, ("undefined" == typeof ace ? at([ct.aceDir + "/ace.js"]) : Promise.resolve()).then((function() {
                            t.loading = !1;
                            var e = ace.edit(t.$el);
                            e.getSession().setMode("ace/mode/javascript"), e.setOptions({
                                readOnly: !0,
                                showLineNumbers: !1,
                                showFoldWidgets: !1,
                                highlightActiveLine: !1,
                                highlightGutterLine: !1
                            }), e.renderer.$cursorLayer.element.style.display = "none", t._editor = e, t.setCode(t.code)
                        }))
                    },
                    methods: {
                        setCode: function(t) {
                            this._editor && (this._editor.setValue(t), this._editor.selection.setSelectionRange({
                                start: {
                                    row: 1,
                                    column: 4
                                },
                                end: {
                                    row: 1,
                                    column: 4
                                }
                            }))
                        }
                    },
                    watch: {
                        code: function(t) {
                            this.setCode(t)
                        }
                    }
                }, Jt, [], !1, null, null, null);
                Xt.options.__file = "src/editor/FullCodePreview.vue";
                const Yt = Xt.exports;
                var Qt = n(913);

                function te(t, e) {
                    const n = Object.create(null),
                        a = t.split(",");
                    for (let t = 0; t < a.length; t++) n[a[t]] = !0;
                    return e ? t => !!n[t.toLowerCase()] : t => !!n[t]
                }
                const ee = te("Infinity,undefined,NaN,isFinite,isNaN,parseFloat,parseInt,decodeURI,decodeURIComponent,encodeURI,encodeURIComponent,Math,Number,Date,Array,Object,Boolean,String,RegExp,Map,Set,JSON,Intl"),
                    ne = te("itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly");

                function ae(t) {
                    if (_e(t)) {
                        const e = {};
                        for (let n = 0; n < t.length; n++) {
                            const a = t[n],
                                i = ae(ke(a) ? re(a) : a);
                            if (i)
                                for (const t in i) e[t] = i[t]
                        }
                        return e
                    }
                    if (Ne(t)) return t
                }
                const ie = /;(?![^(]*\))/g,
                    oe = /:(.+)/;

                function re(t) {
                    const e = {};
                    return t.split(ie).forEach((t => {
                        if (t) {
                            const n = t.split(oe);
                            n.length > 1 && (e[n[0].trim()] = n[1].trim())
                        }
                    })), e
                }

                function le(t) {
                    let e = "";
                    if (ke(t)) e = t;
                    else if (_e(t))
                        for (let n = 0; n < t.length; n++) e += le(t[n]) + " ";
                    else if (Ne(t))
                        for (const n in t) t[n] && (e += n + " ");
                    return e.trim()
                }
                const se = t => null == t ? "" : Ne(t) ? JSON.stringify(t, ce, 2) : String(t),
                    ce = (t, e) => Le(e) ? {
                        [`Map(${e.size})`]: [...e.entries()].reduce(((t, [e, n]) => (t[`${e} =>`] = n, t)), {})
                    } : we(e) ? {
                        [`Set(${e.size})`]: [...e.values()]
                    } : !Ne(e) || _e(e) || De(e) ? e : String(e),
                    ue = {},
                    de = [],
                    pe = () => {},
                    fe = () => !1,
                    ge = /^on[^a-z]/,
                    me = t => ge.test(t),
                    he = t => t.startsWith("onUpdate:"),
                    Ce = Object.assign,
                    ye = (t, e) => {
                        const n = t.indexOf(e);
                        n > -1 && t.splice(n, 1)
                    },
                    ve = Object.prototype.hasOwnProperty,
                    be = (t, e) => ve.call(t, e),
                    _e = Array.isArray,
                    Le = t => "[object Map]" === Oe(t),
                    we = t => "[object Set]" === Oe(t),
                    xe = t => "function" == typeof t,
                    ke = t => "string" == typeof t,
                    Se = t => "symbol" == typeof t,
                    Ne = t => null !== t && "object" == typeof t,
                    Me = t => Ne(t) && xe(t.then) && xe(t.catch),
                    Te = Object.prototype.toString,
                    Oe = t => Te.call(t),
                    De = t => "[object Object]" === Oe(t),
                    Ae = t => ke(t) && "NaN" !== t && "-" !== t[0] && "" + parseInt(t, 10) === t,
                    Ee = te(",key,ref,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),
                    Pe = t => {
                        const e = Object.create(null);
                        return n => e[n] || (e[n] = t(n))
                    },
                    Fe = /-(\w)/g,
                    Re = Pe((t => t.replace(Fe, ((t, e) => e ? e.toUpperCase() : "")))),
                    Ie = /\B([A-Z])/g,
                    Be = Pe((t => t.replace(Ie, "-$1").toLowerCase())),
                    je = Pe((t => t.charAt(0).toUpperCase() + t.slice(1))),
                    Ge = Pe((t => t ? `on${je(t)}` : "")),
                    ze = (t, e) => t !== e && (t == t || e == e),
                    $e = (t, e) => {
                        for (let n = 0; n < t.length; n++) t[n](e)
                    },
                    Ue = (t, e, n) => {
                        Object.defineProperty(t, e, {
                            configurable: !0,
                            enumerable: !1,
                            value: n
                        })
                    },
                    Ze = t => {
                        const e = parseFloat(t);
                        return isNaN(e) ? t : e
                    };
                let Ve;
                const We = () => Ve || (Ve = "undefined" != typeof globalThis ? globalThis : "undefined" != typeof self ? self : "undefined" != typeof window ? window : void 0 !== n.g ? n.g : {}),
                    He = new WeakMap,
                    qe = [];
                let Ke;
                const Je = Symbol(""),
                    Xe = Symbol("");

                function Ye(t, e = ue) {
                    (function(t) {
                        return t && !0 === t._isEffect
                    })(t) && (t = t.raw);
                    const n = function(t, e) {
                        const n = function() {
                            if (!n.active) return e.scheduler ? void 0 : t();
                            if (!qe.includes(n)) {
                                en(n);
                                try {
                                    return an.push(nn), nn = !0, qe.push(n), Ke = n, t()
                                } finally {
                                    qe.pop(), rn(), Ke = qe[qe.length - 1]
                                }
                            }
                        };
                        return n.id = tn++, n.allowRecurse = !!e.allowRecurse, n._isEffect = !0, n.active = !0, n.raw = t, n.deps = [], n.options = e, n
                    }(t, e);
                    return e.lazy || n(), n
                }

                function Qe(t) {
                    t.active && (en(t), t.options.onStop && t.options.onStop(), t.active = !1)
                }
                let tn = 0;

                function en(t) {
                    const {
                        deps: e
                    } = t;
                    if (e.length) {
                        for (let n = 0; n < e.length; n++) e[n].delete(t);
                        e.length = 0
                    }
                }
                let nn = !0;
                const an = [];

                function on() {
                    an.push(nn), nn = !1
                }

                function rn() {
                    const t = an.pop();
                    nn = void 0 === t || t
                }

                function ln(t, e, n) {
                    if (!nn || void 0 === Ke) return;
                    let a = He.get(t);
                    a || He.set(t, a = new Map);
                    let i = a.get(n);
                    i || a.set(n, i = new Set), i.has(Ke) || (i.add(Ke), Ke.deps.push(i))
                }

                function sn(t, e, n, a, i, o) {
                    const r = He.get(t);
                    if (!r) return;
                    const l = new Set,
                        s = t => {
                            t && t.forEach((t => {
                                (t !== Ke || t.allowRecurse) && l.add(t)
                            }))
                        };
                    if ("clear" === e) r.forEach(s);
                    else if ("length" === n && _e(t)) r.forEach(((t, e) => {
                        ("length" === e || e >= a) && s(t)
                    }));
                    else switch (void 0 !== n && s(r.get(n)), e) {
                        case "add":
                            _e(t) ? Ae(n) && s(r.get("length")) : (s(r.get(Je)), Le(t) && s(r.get(Xe)));
                            break;
                        case "delete":
                            _e(t) || (s(r.get(Je)), Le(t) && s(r.get(Xe)));
                            break;
                        case "set":
                            Le(t) && s(r.get(Je))
                    }
                    l.forEach((t => {
                        t.options.scheduler ? t.options.scheduler(t) : t()
                    }))
                }
                const cn = new Set(Object.getOwnPropertyNames(Symbol).map((t => Symbol[t])).filter(Se)),
                    un = mn(),
                    dn = mn(!1, !0),
                    pn = mn(!0),
                    fn = mn(!0, !0),
                    gn = {};

                function mn(t = !1, e = !1) {
                    return function(n, a, i) {
                        if ("__v_isReactive" === a) return !t;
                        if ("__v_isReadonly" === a) return t;
                        if ("__v_raw" === a && i === (t ? $n : zn).get(n)) return n;
                        const o = _e(n);
                        if (!t && o && be(gn, a)) return Reflect.get(gn, a, i);
                        const r = Reflect.get(n, a, i);
                        return (Se(a) ? cn.has(a) : "__proto__" === a || "__v_isRef" === a) ? r : (t || ln(n, 0, a), e ? r : Xn(r) ? o && Ae(a) ? r : r.value : Ne(r) ? t ? Zn(r) : Un(r) : r)
                    }
                }

                function hn(t = !1) {
                    return function(e, n, a, i) {
                        const o = e[n];
                        if (!t && (a = Kn(a), !_e(e) && Xn(o) && !Xn(a))) return o.value = a, !0;
                        const r = _e(e) && Ae(n) ? Number(n) < e.length : be(e, n),
                            l = Reflect.set(e, n, a, i);
                        return e === Kn(i) && (r ? ze(a, o) && sn(e, "set", n, a) : sn(e, "add", n, a)), l
                    }
                }["includes", "indexOf", "lastIndexOf"].forEach((t => {
                    const e = Array.prototype[t];
                    gn[t] = function(...t) {
                        const n = Kn(this);
                        for (let t = 0, e = this.length; t < e; t++) ln(n, 0, t + "");
                        const a = e.apply(n, t);
                        return -1 === a || !1 === a ? e.apply(n, t.map(Kn)) : a
                    }
                })), ["push", "pop", "shift", "unshift", "splice"].forEach((t => {
                    const e = Array.prototype[t];
                    gn[t] = function(...t) {
                        on();
                        const n = e.apply(this, t);
                        return rn(), n
                    }
                }));
                const Cn = {
                        get: un,
                        set: hn(),
                        deleteProperty: function(t, e) {
                            const n = be(t, e),
                                a = (t[e], Reflect.deleteProperty(t, e));
                            return a && n && sn(t, "delete", e, void 0), a
                        },
                        has: function(t, e) {
                            const n = Reflect.has(t, e);
                            return Se(e) && cn.has(e) || ln(t, 0, e), n
                        },
                        ownKeys: function(t) {
                            return ln(t, 0, _e(t) ? "length" : Je), Reflect.ownKeys(t)
                        }
                    },
                    yn = {
                        get: pn,
                        set: (t, e) => !0,
                        deleteProperty: (t, e) => !0
                    },
                    vn = Ce({}, Cn, {
                        get: dn,
                        set: hn(!0)
                    }),
                    bn = (Ce({}, yn, {
                        get: fn
                    }), t => Ne(t) ? Un(t) : t),
                    _n = t => Ne(t) ? Zn(t) : t,
                    Ln = t => t,
                    wn = t => Reflect.getPrototypeOf(t);

                function xn(t, e, n = !1, a = !1) {
                    const i = Kn(t = t.__v_raw),
                        o = Kn(e);
                    e !== o && !n && ln(i, 0, e), !n && ln(i, 0, o);
                    const {
                        has: r
                    } = wn(i), l = n ? _n : a ? Ln : bn;
                    return r.call(i, e) ? l(t.get(e)) : r.call(i, o) ? l(t.get(o)) : void 0
                }

                function kn(t, e = !1) {
                    const n = this.__v_raw,
                        a = Kn(n),
                        i = Kn(t);
                    return t !== i && !e && ln(a, 0, t), !e && ln(a, 0, i), t === i ? n.has(t) : n.has(t) || n.has(i)
                }

                function Sn(t, e = !1) {
                    return t = t.__v_raw, !e && ln(Kn(t), 0, Je), Reflect.get(t, "size", t)
                }

                function Nn(t) {
                    t = Kn(t);
                    const e = Kn(this),
                        n = wn(e).has.call(e, t);
                    return e.add(t), n || sn(e, "add", t, t), this
                }

                function Mn(t, e) {
                    e = Kn(e);
                    const n = Kn(this),
                        {
                            has: a,
                            get: i
                        } = wn(n);
                    let o = a.call(n, t);
                    o || (t = Kn(t), o = a.call(n, t));
                    const r = i.call(n, t);
                    return n.set(t, e), o ? ze(e, r) && sn(n, "set", t, e) : sn(n, "add", t, e), this
                }

                function Tn(t) {
                    const e = Kn(this),
                        {
                            has: n,
                            get: a
                        } = wn(e);
                    let i = n.call(e, t);
                    i || (t = Kn(t), i = n.call(e, t)), a && a.call(e, t);
                    const o = e.delete(t);
                    return i && sn(e, "delete", t, void 0), o
                }

                function On() {
                    const t = Kn(this),
                        e = 0 !== t.size,
                        n = t.clear();
                    return e && sn(t, "clear", void 0, void 0), n
                }

                function Dn(t, e) {
                    return function(n, a) {
                        const i = this,
                            o = i.__v_raw,
                            r = Kn(o),
                            l = t ? _n : e ? Ln : bn;
                        return !t && ln(r, 0, Je), o.forEach(((t, e) => n.call(a, l(t), l(e), i)))
                    }
                }

                function An(t, e, n) {
                    return function(...a) {
                        const i = this.__v_raw,
                            o = Kn(i),
                            r = Le(o),
                            l = "entries" === t || t === Symbol.iterator && r,
                            s = "keys" === t && r,
                            c = i[t](...a),
                            u = e ? _n : n ? Ln : bn;
                        return !e && ln(o, 0, s ? Xe : Je), {
                            next() {
                                const {
                                    value: t,
                                    done: e
                                } = c.next();
                                return e ? {
                                    value: t,
                                    done: e
                                } : {
                                    value: l ? [u(t[0]), u(t[1])] : u(t),
                                    done: e
                                }
                            },
                            [Symbol.iterator]() {
                                return this
                            }
                        }
                    }
                }

                function En(t) {
                    return function(...e) {
                        return "delete" !== t && this
                    }
                }
                const Pn = {
                        get(t) {
                            return xn(this, t)
                        },
                        get size() {
                            return Sn(this)
                        },
                        has: kn,
                        add: Nn,
                        set: Mn,
                        delete: Tn,
                        clear: On,
                        forEach: Dn(!1, !1)
                    },
                    Fn = {
                        get(t) {
                            return xn(this, t, !1, !0)
                        },
                        get size() {
                            return Sn(this)
                        },
                        has: kn,
                        add: Nn,
                        set: Mn,
                        delete: Tn,
                        clear: On,
                        forEach: Dn(!1, !0)
                    },
                    Rn = {
                        get(t) {
                            return xn(this, t, !0)
                        },
                        get size() {
                            return Sn(this, !0)
                        },
                        has(t) {
                            return kn.call(this, t, !0)
                        },
                        add: En("add"),
                        set: En("set"),
                        delete: En("delete"),
                        clear: En("clear"),
                        forEach: Dn(!0, !1)
                    };

                function In(t, e) {
                    const n = e ? Fn : t ? Rn : Pn;
                    return (e, a, i) => "__v_isReactive" === a ? !t : "__v_isReadonly" === a ? t : "__v_raw" === a ? e : Reflect.get(be(n, a) && a in e ? n : e, a, i)
                }["keys", "values", "entries", Symbol.iterator].forEach((t => {
                    Pn[t] = An(t, !1, !1), Rn[t] = An(t, !0, !1), Fn[t] = An(t, !1, !0)
                }));
                const Bn = {
                        get: In(!1, !1)
                    },
                    jn = {
                        get: In(!1, !0)
                    },
                    Gn = {
                        get: In(!0, !1)
                    },
                    zn = new WeakMap,
                    $n = new WeakMap;

                function Un(t) {
                    return t && t.__v_isReadonly ? t : Vn(t, !1, Cn, Bn)
                }

                function Zn(t) {
                    return Vn(t, !0, yn, Gn)
                }

                function Vn(t, e, n, a) {
                    if (!Ne(t)) return t;
                    if (t.__v_raw && (!e || !t.__v_isReactive)) return t;
                    const i = e ? $n : zn,
                        o = i.get(t);
                    if (o) return o;
                    const r = function(t) {
                        return t.__v_skip || !Object.isExtensible(t) ? 0 : function(t) {
                            switch (t) {
                                case "Object":
                                case "Array":
                                    return 1;
                                case "Map":
                                case "Set":
                                case "WeakMap":
                                case "WeakSet":
                                    return 2;
                                default:
                                    return 0
                            }
                        }((t => Oe(t).slice(8, -1))(t))
                    }(t);
                    if (0 === r) return t;
                    const l = new Proxy(t, 2 === r ? a : n);
                    return i.set(t, l), l
                }

                function Wn(t) {
                    return Hn(t) ? Wn(t.__v_raw) : !(!t || !t.__v_isReactive)
                }

                function Hn(t) {
                    return !(!t || !t.__v_isReadonly)
                }

                function qn(t) {
                    return Wn(t) || Hn(t)
                }

                function Kn(t) {
                    return t && Kn(t.__v_raw) || t
                }
                const Jn = t => Ne(t) ? Un(t) : t;

                function Xn(t) {
                    return Boolean(t && !0 === t.__v_isRef)
                }

                function Yn(t) {
                    return function(t, e = !1) {
                        return Xn(t) ? t : new Qn(t, e)
                    }(t)
                }
                class Qn {
                    constructor(t, e = !1) {
                        this._rawValue = t, this._shallow = e, this.__v_isRef = !0, this._value = e ? t : Jn(t)
                    }
                    get value() {
                        return ln(Kn(this), 0, "value"), this._value
                    }
                    set value(t) {
                        ze(Kn(t), this._rawValue) && (this._rawValue = t, this._value = this._shallow ? t : Jn(t), sn(Kn(this), "set", "value", t))
                    }
                }
                const ta = {
                    get: (t, e, n) => function(t) {
                        return Xn(t) ? t.value : t
                    }(Reflect.get(t, e, n)),
                    set: (t, e, n, a) => {
                        const i = t[e];
                        return Xn(i) && !Xn(n) ? (i.value = n, !0) : Reflect.set(t, e, n, a)
                    }
                };

                function ea(t) {
                    return Wn(t) ? t : new Proxy(t, ta)
                }
                class na {
                    constructor(t, e) {
                        this._object = t, this._key = e, this.__v_isRef = !0
                    }
                    get value() {
                        return this._object[this._key]
                    }
                    set value(t) {
                        this._object[this._key] = t
                    }
                }
                class aa {
                    constructor(t, e, n) {
                        this._setter = e, this._dirty = !0, this.__v_isRef = !0, this.effect = Ye(t, {
                            lazy: !0,
                            scheduler: () => {
                                this._dirty || (this._dirty = !0, sn(Kn(this), "set", "value"))
                            }
                        }), this.__v_isReadonly = n
                    }
                    get value() {
                        return this._dirty && (this._value = this.effect(), this._dirty = !1), ln(Kn(this), 0, "value"), this._value
                    }
                    set value(t) {
                        this._setter(t)
                    }
                }

                function ia(t, e, n, a) {
                    let i;
                    try {
                        i = a ? t(...a) : t()
                    } catch (t) {
                        ra(t, e, n)
                    }
                    return i
                }

                function oa(t, e, n, a) {
                    if (xe(t)) {
                        const i = ia(t, e, n, a);
                        return i && Me(i) && i.catch((t => {
                            ra(t, e, n)
                        })), i
                    }
                    const i = [];
                    for (let o = 0; o < t.length; o++) i.push(oa(t[o], e, n, a));
                    return i
                }

                function ra(t, e, n, a = !0) {
                    if (e && e.vnode, e) {
                        let a = e.parent;
                        const i = e.proxy,
                            o = n;
                        for (; a;) {
                            const e = a.ec;
                            if (e)
                                for (let n = 0; n < e.length; n++)
                                    if (!1 === e[n](t, i, o)) return;
                            a = a.parent
                        }
                        const r = e.appContext.config.errorHandler;
                        if (r) return void ia(r, null, 10, [t, i, o])
                    }! function(t, e, n, a = !0) {
                        console.error(t)
                    }(t, 0, 0, a)
                }
                let la = !1,
                    sa = !1;
                const ca = [];
                let ua = 0;
                const da = [];
                let pa = null,
                    fa = 0;
                const ga = [];
                let ma = null,
                    ha = 0;
                const Ca = Promise.resolve();
                let ya = null,
                    va = null;

                function ba(t) {
                    const e = ya || Ca;
                    return t ? e.then(this ? t.bind(this) : t) : e
                }

                function _a(t) {
                    ca.length && ca.includes(t, la && t.allowRecurse ? ua + 1 : ua) || t === va || (ca.push(t), La())
                }

                function La() {
                    la || sa || (sa = !0, ya = Ca.then(Na))
                }

                function wa(t, e, n, a) {
                    _e(t) ? n.push(...t) : e && e.includes(t, t.allowRecurse ? a + 1 : a) || n.push(t), La()
                }

                function xa(t, e = null) {
                    if (da.length) {
                        for (va = e, pa = [...new Set(da)], da.length = 0, fa = 0; fa < pa.length; fa++) pa[fa]();
                        pa = null, fa = 0, va = null, xa(t, e)
                    }
                }

                function ka(t) {
                    if (ga.length) {
                        const t = [...new Set(ga)];
                        if (ga.length = 0, ma) return void ma.push(...t);
                        for (ma = t, ma.sort(((t, e) => Sa(t) - Sa(e))), ha = 0; ha < ma.length; ha++) ma[ha]();
                        ma = null, ha = 0
                    }
                }
                const Sa = t => null == t.id ? 1 / 0 : t.id;

                function Na(t) {
                    sa = !1, la = !0, xa(t), ca.sort(((t, e) => Sa(t) - Sa(e)));
                    try {
                        for (ua = 0; ua < ca.length; ua++) {
                            const t = ca[ua];
                            t && ia(t, null, 14)
                        }
                    } finally {
                        ua = 0, ca.length = 0, ka(), la = !1, ya = null, (ca.length || ga.length) && Na(t)
                    }
                }

                function Ma(t, e, ...n) {
                    const a = t.vnode.props || ue;
                    let i = n;
                    const o = e.startsWith("update:"),
                        r = o && e.slice(7);
                    if (r && r in a) {
                        const t = `${"modelValue"===r?"model":r}Modifiers`,
                            {
                                number: e,
                                trim: o
                            } = a[t] || ue;
                        o ? i = n.map((t => t.trim())) : e && (i = n.map(Ze))
                    }
                    __VUE_PROD_DEVTOOLS__;
                    let l = Ge(Re(e)),
                        s = a[l];
                    !s && o && (l = Ge(Be(e)), s = a[l]), s && oa(s, t, 6, i);
                    const c = a[l + "Once"];
                    if (c) {
                        if (t.emitted) {
                            if (t.emitted[l]) return
                        } else(t.emitted = {})[l] = !0;
                        oa(c, t, 6, i)
                    }
                }

                function Ta(t, e, n = !1) {
                    if (!e.deopt && void 0 !== t.__emits) return t.__emits;
                    const a = t.emits;
                    let i = {},
                        o = !1;
                    if (__VUE_OPTIONS_API__ && !xe(t)) {
                        const a = t => {
                            o = !0, Ce(i, Ta(t, e, !0))
                        };
                        !n && e.mixins.length && e.mixins.forEach(a), t.extends && a(t.extends), t.mixins && t.mixins.forEach(a)
                    }
                    return a || o ? (_e(a) ? a.forEach((t => i[t] = null)) : Ce(i, a), t.__emits = i) : t.__emits = null
                }

                function Oa(t, e) {
                    return !(!t || !me(e)) && (e = e.slice(2).replace(/Once$/, ""), be(t, e[0].toLowerCase() + e.slice(1)) || be(t, Be(e)) || be(t, e))
                }
                let Da = null;

                function Aa(t) {
                    Da = t
                }

                function Ea(t) {
                    const {
                        type: e,
                        vnode: n,
                        proxy: a,
                        withProxy: i,
                        props: o,
                        propsOptions: [r],
                        slots: l,
                        attrs: s,
                        emit: c,
                        render: u,
                        renderCache: d,
                        data: p,
                        setupState: f,
                        ctx: g
                    } = t;
                    let m;
                    Da = t;
                    try {
                        let t;
                        if (4 & n.shapeFlag) {
                            const e = i || a;
                            m = qi(u.call(e, e, d, o, f, p, g)), t = s
                        } else {
                            const n = e;
                            m = qi(n.length > 1 ? n(o, {
                                attrs: s,
                                slots: l,
                                emit: c
                            }) : n(o, null)), t = e.props ? s : Pa(s)
                        }
                        let h = m;
                        if (!1 !== e.inheritAttrs && t) {
                            const e = Object.keys(t),
                                {
                                    shapeFlag: n
                                } = h;
                            e.length && (1 & n || 6 & n) && (r && e.some(he) && (t = Fa(t, r)), h = Vi(h, t))
                        }
                        n.dirs && (h.dirs = h.dirs ? h.dirs.concat(n.dirs) : n.dirs), n.transition && (h.transition = n.transition), m = h
                    } catch (e) {
                        ra(e, t, 1), m = Zi(Ai)
                    }
                    return Da = null, m
                }
                const Pa = t => {
                        let e;
                        for (const n in t)("class" === n || "style" === n || me(n)) && ((e || (e = {}))[n] = t[n]);
                        return e
                    },
                    Fa = (t, e) => {
                        const n = {};
                        for (const a in t) he(a) && a.slice(9) in e || (n[a] = t[a]);
                        return n
                    };

                function Ra(t, e, n) {
                    const a = Object.keys(e);
                    if (a.length !== Object.keys(t).length) return !0;
                    for (let i = 0; i < a.length; i++) {
                        const o = a[i];
                        if (e[o] !== t[o] && !Oa(n, o)) return !0
                    }
                    return !1
                }

                function Ia(t) {
                    return xe(t) && (t = t()), _e(t) && (t = function(t) {
                        let e;
                        for (let n = 0; n < t.length; n++) {
                            const a = t[n];
                            if (!ji(a)) return;
                            if (a.type !== Ai || "v-if" === a.children) {
                                if (e) return;
                                e = a
                            }
                        }
                        return e
                    }(t)), qi(t)
                }
                let Ba = 0;
                const ja = t => Ba += t;

                function Ga(t, e, n, a) {
                    const [i, o] = t.propsOptions;
                    if (e)
                        for (const o in e) {
                            const r = e[o];
                            if (Ee(o)) continue;
                            let l;
                            i && be(i, l = Re(o)) ? n[l] = r : Oa(t.emitsOptions, o) || (a[o] = r)
                        }
                    if (o) {
                        const e = Kn(n);
                        for (let a = 0; a < o.length; a++) {
                            const r = o[a];
                            n[r] = za(i, e, r, e[r], t)
                        }
                    }
                }

                function za(t, e, n, a, i) {
                    const o = t[n];
                    if (null != o) {
                        const t = be(o, "default");
                        if (t && void 0 === a) {
                            const t = o.default;
                            o.type !== Function && xe(t) ? (mo(i), a = t(e), mo(null)) : a = t
                        }
                        o[0] && (be(e, n) || t ? !o[1] || "" !== a && a !== Be(n) || (a = !0) : a = !1)
                    }
                    return a
                }

                function $a(t, e, n = !1) {
                    if (!e.deopt && t.__props) return t.__props;
                    const a = t.props,
                        i = {},
                        o = [];
                    let r = !1;
                    if (__VUE_OPTIONS_API__ && !xe(t)) {
                        const a = t => {
                            r = !0;
                            const [n, a] = $a(t, e, !0);
                            Ce(i, n), a && o.push(...a)
                        };
                        !n && e.mixins.length && e.mixins.forEach(a), t.extends && a(t.extends), t.mixins && t.mixins.forEach(a)
                    }
                    if (!a && !r) return t.__props = de;
                    if (_e(a))
                        for (let t = 0; t < a.length; t++) {
                            const e = Re(a[t]);
                            Ua(e) && (i[e] = ue)
                        } else if (a)
                            for (const t in a) {
                                const e = Re(t);
                                if (Ua(e)) {
                                    const n = a[t],
                                        r = i[e] = _e(n) || xe(n) ? {
                                            type: n
                                        } : n;
                                    if (r) {
                                        const t = Wa(Boolean, r.type),
                                            n = Wa(String, r.type);
                                        r[0] = t > -1, r[1] = n < 0 || t < n, (t > -1 || be(r, "default")) && o.push(e)
                                    }
                                }
                            }
                    return t.__props = [i, o]
                }

                function Ua(t) {
                    return "$" !== t[0]
                }

                function Za(t) {
                    const e = t && t.toString().match(/^\s*function (\w+)/);
                    return e ? e[1] : ""
                }

                function Va(t, e) {
                    return Za(t) === Za(e)
                }

                function Wa(t, e) {
                    if (_e(e)) {
                        for (let n = 0, a = e.length; n < a; n++)
                            if (Va(e[n], t)) return n
                    } else if (xe(e)) return Va(e, t) ? 0 : -1;
                    return -1
                }

                function Ha(t, e, n = go, a = !1) {
                    if (n) {
                        const i = n[t] || (n[t] = []),
                            o = e.__weh || (e.__weh = (...a) => {
                                if (n.isUnmounted) return;
                                on(), mo(n);
                                const i = oa(e, n, t, a);
                                return mo(null), rn(), i
                            });
                        return a ? i.unshift(o) : i.push(o), o
                    }
                }
                const qa = t => (e, n = go) => !ho && Ha(t, e, n),
                    Ka = qa("bm"),
                    Ja = qa("m"),
                    Xa = qa("bu"),
                    Ya = qa("u"),
                    Qa = qa("bum"),
                    ti = qa("um"),
                    ei = qa("rtg"),
                    ni = qa("rtc"),
                    ai = {};

                function ii(t, e, n) {
                    return oi(t, e, n)
                }

                function oi(t, e, {
                    immediate: n,
                    deep: a,
                    flush: i,
                    onTrack: o,
                    onTrigger: r
                } = ue, l = go) {
                    let s, c, u = !1;
                    if (Xn(t) ? (s = () => t.value, u = !!t._shallow) : Wn(t) ? (s = () => t, a = !0) : s = _e(t) ? () => t.map((t => Xn(t) ? t.value : Wn(t) ? li(t) : xe(t) ? ia(t, l, 2) : void 0)) : xe(t) ? e ? () => ia(t, l, 2) : () => {
                            if (!l || !l.isUnmounted) return c && c(), ia(t, l, 3, [d])
                        } : pe, e && a) {
                        const t = s;
                        s = () => li(t())
                    }
                    const d = t => {
                        c = m.options.onStop = () => {
                            ia(t, l, 4)
                        }
                    };
                    let p = _e(t) ? [] : ai;
                    const f = () => {
                        if (m.active)
                            if (e) {
                                const t = m();
                                (a || u || ze(t, p)) && (c && c(), oa(e, l, 3, [t, p === ai ? void 0 : p, d]), p = t)
                            } else m()
                    };
                    let g;
                    f.allowRecurse = !!e, g = "sync" === i ? f : "post" === i ? () => Li(f, l && l.suspense) : () => {
                        !l || l.isMounted ? function(t) {
                            wa(t, pa, da, fa)
                        }(f) : f()
                    };
                    const m = Ye(s, {
                        lazy: !0,
                        onTrack: o,
                        onTrigger: r,
                        scheduler: g
                    });
                    return vo(m, l), e ? n ? f() : p = m() : "post" === i ? Li(m, l && l.suspense) : m(), () => {
                        Qe(m), l && ye(l.effects, m)
                    }
                }

                function ri(t, e, n) {
                    const a = this.proxy;
                    return oi(ke(t) ? () => a[t] : t.bind(a), e.bind(a), n, this)
                }

                function li(t, e = new Set) {
                    if (!Ne(t) || e.has(t)) return t;
                    if (e.add(t), Xn(t)) li(t.value, e);
                    else if (_e(t))
                        for (let n = 0; n < t.length; n++) li(t[n], e);
                    else if (we(t) || Le(t)) t.forEach((t => {
                        li(t, e)
                    }));
                    else
                        for (const n in t) li(t[n], e);
                    return t
                }
                const si = t => t.type.__isKeepAlive;

                function ci(t, e, n = go) {
                    const a = t.__wdc || (t.__wdc = () => {
                        let e = n;
                        for (; e;) {
                            if (e.isDeactivated) return;
                            e = e.parent
                        }
                        t()
                    });
                    if (Ha(e, a, n), n) {
                        let t = n.parent;
                        for (; t && t.parent;) si(t.parent.vnode) && ui(a, e, n, t), t = t.parent
                    }
                }

                function ui(t, e, n, a) {
                    const i = Ha(e, t, a, !0);
                    ti((() => {
                        ye(a[e], i)
                    }), n)
                }
                const di = t => "_" === t[0] || "$stable" === t,
                    pi = t => _e(t) ? t.map(qi) : [qi(t)],
                    fi = (t, e, n) => function(t, e = Da) {
                        if (!e) return t;
                        const n = (...n) => {
                            Ba || Ri(!0);
                            const a = Da;
                            Aa(e);
                            const i = t(...n);
                            return Aa(a), Ba || Ii(), i
                        };
                        return n._c = !0, n
                    }((t => pi(e(t))), n),
                    gi = (t, e) => {
                        const n = t._ctx;
                        for (const a in t) {
                            if (di(a)) continue;
                            const i = t[a];
                            if (xe(i)) e[a] = fi(0, i, n);
                            else if (null != i) {
                                const t = pi(i);
                                e[a] = () => t
                            }
                        }
                    },
                    mi = (t, e) => {
                        const n = pi(e);
                        t.slots.default = () => n
                    };

                function hi(t, e) {
                    if (null === Da) return t;
                    const n = Da.proxy,
                        a = t.dirs || (t.dirs = []);
                    for (let t = 0; t < e.length; t++) {
                        let [i, o, r, l = ue] = e[t];
                        xe(i) && (i = {
                            mounted: i,
                            updated: i
                        }), a.push({
                            dir: i,
                            instance: n,
                            value: o,
                            oldValue: void 0,
                            arg: r,
                            modifiers: l
                        })
                    }
                    return t
                }

                function Ci(t, e, n, a) {
                    const i = t.dirs,
                        o = e && e.dirs;
                    for (let r = 0; r < i.length; r++) {
                        const l = i[r];
                        o && (l.oldValue = o[r].value);
                        const s = l.dir[a];
                        s && oa(s, n, 8, [t.el, l, t, e])
                    }
                }

                function yi() {
                    return {
                        app: null,
                        config: {
                            isNativeTag: fe,
                            performance: !1,
                            globalProperties: {},
                            optionMergeStrategies: {},
                            isCustomElement: fe,
                            errorHandler: void 0,
                            warnHandler: void 0
                        },
                        mixins: [],
                        components: {},
                        directives: {},
                        provides: Object.create(null)
                    }
                }
                let vi = 0;

                function bi(t, e) {
                    return function(n, a = null) {
                        null == a || Ne(a) || (a = null);
                        const i = yi(),
                            o = new Set;
                        let r = !1;
                        const l = i.app = {
                            _uid: vi++,
                            _component: n,
                            _props: a,
                            _container: null,
                            _context: i,
                            version: Lo,
                            get config() {
                                return i.config
                            },
                            set config(t) {},
                            use: (t, ...e) => (o.has(t) || (t && xe(t.install) ? (o.add(t), t.install(l, ...e)) : xe(t) && (o.add(t), t(l, ...e))), l),
                            mixin: t => (__VUE_OPTIONS_API__ && (i.mixins.includes(t) || (i.mixins.push(t), (t.props || t.emits) && (i.deopt = !0))), l),
                            component: (t, e) => e ? (i.components[t] = e, l) : i.components[t],
                            directive: (t, e) => e ? (i.directives[t] = e, l) : i.directives[t],
                            mount(o, s) {
                                if (!r) {
                                    const c = Zi(n, a);
                                    return c.appContext = i, s && e ? e(c, o) : t(c, o), r = !0, l._container = o, o.__vue_app__ = l, __VUE_PROD_DEVTOOLS__, c.component.proxy
                                }
                            },
                            unmount() {
                                r && (t(null, l._container), __VUE_PROD_DEVTOOLS__)
                            },
                            provide: (t, e) => (i.provides[t] = e, l)
                        };
                        return l
                    }
                }
                const _i = {
                        scheduler: _a,
                        allowRecurse: !0
                    },
                    Li = function(t, e) {
                        e && e.pendingBranch ? _e(t) ? e.effects.push(...t) : e.effects.push(t) : wa(t, ma, ga, ha)
                    },
                    wi = (t, e, n, a) => {
                        if (_e(t)) return void t.forEach(((t, i) => wi(t, e && (_e(e) ? e[i] : e), n, a)));
                        let i;
                        i = !a || a.type.__asyncLoader ? null : 4 & a.shapeFlag ? a.component.exposed || a.component.proxy : a.el;
                        const {
                            i: o,
                            r
                        } = t, l = e && e.r, s = o.refs === ue ? o.refs = {} : o.refs, c = o.setupState;
                        if (null != l && l !== r && (ke(l) ? (s[l] = null, be(c, l) && (c[l] = null)) : Xn(l) && (l.value = null)), ke(r)) {
                            const t = () => {
                                s[r] = i, be(c, r) && (c[r] = i)
                            };
                            i ? (t.id = -1, Li(t, n)) : t()
                        } else if (Xn(r)) {
                            const t = () => {
                                r.value = i
                            };
                            i ? (t.id = -1, Li(t, n)) : t()
                        } else xe(r) && ia(r, o, 12, [i, s])
                    };

                function xi(t, e, n, a = null) {
                    oa(t, e, 7, [n, a])
                }

                function ki(t, e, n = !1) {
                    const a = t.children,
                        i = e.children;
                    if (_e(a) && _e(i))
                        for (let t = 0; t < a.length; t++) {
                            const e = a[t];
                            let o = i[t];
                            1 & o.shapeFlag && !o.dynamicChildren && ((o.patchFlag <= 0 || 32 === o.patchFlag) && (o = i[t] = Ki(i[t]), o.el = e.el), n || ki(e, o))
                        }
                }
                const Si = t => t && (t.disabled || "" === t.disabled);

                function Ni(t) {
                    return function(t, e, n = !0) {
                        const a = Da || go;
                        if (a) {
                            const n = a.type;
                            if ("components" === t) {
                                if ("_self" === e) return n;
                                const t = n.displayName || n.name;
                                if (t && (t === e || t === Re(e) || t === je(Re(e)))) return n
                            }
                            return Ti(a[t] || n[t], e) || Ti(a.appContext[t], e)
                        }
                    }("components", t) || t
                }
                const Mi = Symbol();

                function Ti(t, e) {
                    return t && (t[e] || t[Re(e)] || t[je(Re(e))])
                }
                const Oi = Symbol(void 0),
                    Di = Symbol(void 0),
                    Ai = Symbol(void 0),
                    Ei = Symbol(void 0),
                    Pi = [];
                let Fi = null;

                function Ri(t = !1) {
                    Pi.push(Fi = t ? null : [])
                }

                function Ii() {
                    Pi.pop(), Fi = Pi[Pi.length - 1] || null
                }

                function Bi(t, e, n, a, i) {
                    const o = Zi(t, e, n, a, i, !0);
                    return o.dynamicChildren = Fi || de, Ii(), Fi && Fi.push(o), o
                }

                function ji(t) {
                    return !!t && !0 === t.__v_isVNode
                }

                function Gi(t, e) {
                    return t.type === e.type && t.key === e.key
                }
                const zi = "__vInternal",
                    $i = ({
                        key: t
                    }) => null != t ? t : null,
                    Ui = ({
                        ref: t
                    }) => null != t ? ke(t) || Xn(t) || xe(t) ? {
                        i: Da,
                        r: t
                    } : t : null,
                    Zi = function(t, e = null, n = null, a = 0, i = null, o = !1) {
                        if (t && t !== Mi || (t = Ai), ji(t)) {
                            const a = Vi(t, e, !0);
                            return n && Ji(a, n), a
                        }
                        var r;
                        if (xe(r = t) && "__vccOpts" in r && (t = t.__vccOpts), e) {
                            (qn(e) || zi in e) && (e = Ce({}, e));
                            let {
                                class: t,
                                style: n
                            } = e;
                            t && !ke(t) && (e.class = le(t)), Ne(n) && (qn(n) && !_e(n) && (n = Ce({}, n)), e.style = ae(n))
                        }
                        const l = ke(t) ? 1 : (t => t.__isSuspense)(t) ? 128 : (t => t.__isTeleport)(t) ? 64 : Ne(t) ? 4 : xe(t) ? 2 : 0,
                            s = {
                                __v_isVNode: !0,
                                __v_skip: !0,
                                type: t,
                                props: e,
                                key: e && $i(e),
                                ref: e && Ui(e),
                                scopeId: null,
                                children: null,
                                component: null,
                                suspense: null,
                                ssContent: null,
                                ssFallback: null,
                                dirs: null,
                                transition: null,
                                el: null,
                                anchor: null,
                                target: null,
                                targetAnchor: null,
                                staticCount: 0,
                                shapeFlag: l,
                                patchFlag: a,
                                dynamicProps: i,
                                dynamicChildren: null,
                                appContext: null
                            };
                        if (Ji(s, n), 128 & l) {
                            const {
                                content: t,
                                fallback: e
                            } = function(t) {
                                const {
                                    shapeFlag: e,
                                    children: n
                                } = t;
                                let a, i;
                                return 32 & e ? (a = Ia(n.default), i = Ia(n.fallback)) : (a = Ia(n), i = qi(null)), {
                                    content: a,
                                    fallback: i
                                }
                            }(s);
                            s.ssContent = t, s.ssFallback = e
                        }
                        return !o && Fi && (a > 0 || 6 & l) && 32 !== a && Fi.push(s), s
                    };

                function Vi(t, e, n = !1) {
                    const {
                        props: a,
                        ref: i,
                        patchFlag: o
                    } = t, r = e ? function(...t) {
                        const e = Ce({}, t[0]);
                        for (let n = 1; n < t.length; n++) {
                            const a = t[n];
                            for (const t in a)
                                if ("class" === t) e.class !== a.class && (e.class = le([e.class, a.class]));
                                else if ("style" === t) e.style = ae([e.style, a.style]);
                            else if (me(t)) {
                                const n = e[t],
                                    i = a[t];
                                n !== i && (e[t] = n ? [].concat(n, a[t]) : i)
                            } else "" !== t && (e[t] = a[t])
                        }
                        return e
                    }(a || {}, e) : a;
                    return {
                        __v_isVNode: !0,
                        __v_skip: !0,
                        type: t.type,
                        props: r,
                        key: r && $i(r),
                        ref: e && e.ref ? n && i ? _e(i) ? i.concat(Ui(e)) : [i, Ui(e)] : Ui(e) : i,
                        scopeId: t.scopeId,
                        children: t.children,
                        target: t.target,
                        targetAnchor: t.targetAnchor,
                        staticCount: t.staticCount,
                        shapeFlag: t.shapeFlag,
                        patchFlag: e && t.type !== Oi ? -1 === o ? 16 : 16 | o : o,
                        dynamicProps: t.dynamicProps,
                        dynamicChildren: t.dynamicChildren,
                        appContext: t.appContext,
                        dirs: t.dirs,
                        transition: t.transition,
                        component: t.component,
                        suspense: t.suspense,
                        ssContent: t.ssContent && Vi(t.ssContent),
                        ssFallback: t.ssFallback && Vi(t.ssFallback),
                        el: t.el,
                        anchor: t.anchor
                    }
                }

                function Wi(t = " ", e = 0) {
                    return Zi(Di, null, t, e)
                }

                function Hi(t = "", e = !1) {
                    return e ? (Ri(), Bi(Ai, null, t)) : Zi(Ai, null, t)
                }

                function qi(t) {
                    return null == t || "boolean" == typeof t ? Zi(Ai) : _e(t) ? Zi(Oi, null, t) : "object" == typeof t ? null === t.el ? t : Vi(t) : Zi(Di, null, String(t))
                }

                function Ki(t) {
                    return null === t.el ? t : Vi(t)
                }

                function Ji(t, e) {
                    let n = 0;
                    const {
                        shapeFlag: a
                    } = t;
                    if (null == e) e = null;
                    else if (_e(e)) n = 16;
                    else if ("object" == typeof e) {
                        if (1 & a || 64 & a) {
                            const n = e.default;
                            return void(n && (n._c && ja(1), Ji(t, n()), n._c && ja(-1)))
                        } {
                            n = 32;
                            const a = e._;
                            a || zi in e ? 3 === a && Da && (1024 & Da.vnode.patchFlag ? (e._ = 2, t.patchFlag |= 1024) : e._ = 1) : e._ctx = Da
                        }
                    } else xe(e) ? (e = {
                        default: e,
                        _ctx: Da
                    }, n = 32) : (e = String(e), 64 & a ? (n = 16, e = [Wi(e)]) : n = 8);
                    t.children = e, t.shapeFlag |= n
                }

                function Xi(t, e, n = !1) {
                    const a = go || Da;
                    if (a) {
                        const i = null == a.parent ? a.vnode.appContext && a.vnode.appContext.provides : a.parent.provides;
                        if (i && t in i) return i[t];
                        if (arguments.length > 1) return n && xe(e) ? e() : e
                    }
                }
                let Yi = !1;

                function Qi(t, e, n = [], a = [], i = [], o = !1) {
                    const {
                        mixins: r,
                        extends: l,
                        data: s,
                        computed: c,
                        methods: u,
                        watch: d,
                        provide: p,
                        inject: f,
                        components: g,
                        directives: m,
                        beforeMount: h,
                        mounted: C,
                        beforeUpdate: y,
                        updated: v,
                        activated: b,
                        deactivated: _,
                        beforeDestroy: L,
                        beforeUnmount: w,
                        destroyed: x,
                        unmounted: k,
                        render: S,
                        renderTracked: N,
                        renderTriggered: M,
                        errorCaptured: T,
                        expose: O
                    } = e, D = t.proxy, A = t.ctx, E = t.appContext.mixins;
                    if (o && S && t.render === pe && (t.render = S), o || (Yi = !0, to("beforeCreate", "bc", e, t, E), Yi = !1, ao(t, E, n, a, i)), l && Qi(t, l, n, a, i, !0), r && ao(t, r, n, a, i), f)
                        if (_e(f))
                            for (let t = 0; t < f.length; t++) {
                                const e = f[t];
                                A[e] = Xi(e)
                            } else
                                for (const t in f) {
                                    const e = f[t];
                                    Ne(e) ? A[t] = Xi(e.from || t, e.default, !0) : A[t] = Xi(e)
                                }
                    if (u)
                        for (const t in u) {
                            const e = u[t];
                            xe(e) && (A[t] = e.bind(D))
                        }
                    if (o ? s && n.push(s) : (n.length && n.forEach((e => io(t, e, D))), s && io(t, s, D)), c)
                        for (const t in c) {
                            const e = c[t],
                                n = bo({
                                    get: xe(e) ? e.bind(D, D) : xe(e.get) ? e.get.bind(D, D) : pe,
                                    set: !xe(e) && xe(e.set) ? e.set.bind(D) : pe
                                });
                            Object.defineProperty(A, t, {
                                enumerable: !0,
                                configurable: !0,
                                get: () => n.value,
                                set: t => n.value = t
                            })
                        }
                    if (d && a.push(d), !o && a.length && a.forEach((t => {
                            for (const e in t) oo(t[e], A, D, e)
                        })), p && i.push(p), !o && i.length && i.forEach((t => {
                            const e = xe(t) ? t.call(D) : t;
                            Reflect.ownKeys(e).forEach((t => {
                                ! function(t, e) {
                                    if (go) {
                                        let n = go.provides;
                                        const a = go.parent && go.parent.provides;
                                        a === n && (n = go.provides = Object.create(a)), n[t] = e
                                    }
                                }(t, e[t])
                            }))
                        })), o && (g && Ce(t.components || (t.components = Ce({}, t.type.components)), g), m && Ce(t.directives || (t.directives = Ce({}, t.type.directives)), m)), o || to("created", "c", e, t, E), h && Ka(h.bind(D)), C && Ja(C.bind(D)), y && Xa(y.bind(D)), v && Ya(v.bind(D)), b && ci(b.bind(D), "a", void 0), _ && function(t, e) {
                            ci(t, "da", void 0)
                        }(_.bind(D)), T && ((t, e = go) => {
                            Ha("ec", t, e)
                        })(T.bind(D)), N && ni(N.bind(D)), M && ei(M.bind(D)), w && Qa(w.bind(D)), k && ti(k.bind(D)), _e(O) && !o)
                        if (O.length) {
                            const e = t.exposed || (t.exposed = ea({}));
                            O.forEach((t => {
                                e[t] = function(t, e) {
                                    return Xn(t[e]) ? t[e] : new na(t, e)
                                }(D, t)
                            }))
                        } else t.exposed || (t.exposed = ue)
                }

                function to(t, e, n, a, i) {
                    no(t, e, i, a);
                    const {
                        extends: o,
                        mixins: r
                    } = n;
                    o && eo(t, e, o, a), r && no(t, e, r, a);
                    const l = n[t];
                    l && oa(l.bind(a.proxy), a, e)
                }

                function eo(t, e, n, a) {
                    n.extends && eo(t, e, n.extends, a);
                    const i = n[t];
                    i && oa(i.bind(a.proxy), a, e)
                }

                function no(t, e, n, a) {
                    for (let i = 0; i < n.length; i++) {
                        const o = n[i].mixins;
                        o && no(t, e, o, a);
                        const r = n[i][t];
                        r && oa(r.bind(a.proxy), a, e)
                    }
                }

                function ao(t, e, n, a, i) {
                    for (let o = 0; o < e.length; o++) Qi(t, e[o], n, a, i, !0)
                }

                function io(t, e, n) {
                    const a = e.call(n, n);
                    Ne(a) && (t.data === ue ? t.data = Un(a) : Ce(t.data, a))
                }

                function oo(t, e, n, a) {
                    const i = a.includes(".") ? function(t, e) {
                        const n = e.split(".");
                        return () => {
                            let e = t;
                            for (let t = 0; t < n.length && e; t++) e = e[n[t]];
                            return e
                        }
                    }(n, a) : () => n[a];
                    if (ke(t)) {
                        const n = e[t];
                        xe(n) && ii(i, n)
                    } else if (xe(t)) ii(i, t.bind(n));
                    else if (Ne(t))
                        if (_e(t)) t.forEach((t => oo(t, e, n, a)));
                        else {
                            const a = xe(t.handler) ? t.handler.bind(n) : e[t.handler];
                            xe(a) && ii(i, a, t)
                        }
                }

                function ro(t, e, n) {
                    const a = n.appContext.config.optionMergeStrategies,
                        {
                            mixins: i,
                            extends: o
                        } = e;
                    o && ro(t, o, n), i && i.forEach((e => ro(t, e, n)));
                    for (const i in e) a && be(a, i) ? t[i] = a[i](t[i], e[i], n.proxy, i) : t[i] = e[i]
                }
                const lo = t => t && (t.proxy ? t.proxy : lo(t.parent)),
                    so = Ce(Object.create(null), {
                        $: t => t,
                        $el: t => t.vnode.el,
                        $data: t => t.data,
                        $props: t => t.props,
                        $attrs: t => t.attrs,
                        $slots: t => t.slots,
                        $refs: t => t.refs,
                        $parent: t => lo(t.parent),
                        $root: t => t.root && t.root.proxy,
                        $emit: t => t.emit,
                        $options: t => __VUE_OPTIONS_API__ ? function(t) {
                            const e = t.type,
                                {
                                    __merged: n,
                                    mixins: a,
                                    extends: i
                                } = e;
                            if (n) return n;
                            const o = t.appContext.mixins;
                            if (!o.length && !a && !i) return e;
                            const r = {};
                            return o.forEach((e => ro(r, e, t))), ro(r, e, t), e.__merged = r
                        }(t) : t.type,
                        $forceUpdate: t => () => _a(t.update),
                        $nextTick: t => ba.bind(t.proxy),
                        $watch: t => __VUE_OPTIONS_API__ ? ri.bind(t) : pe
                    }),
                    co = {
                        get({
                            _: t
                        }, e) {
                            const {
                                ctx: n,
                                setupState: a,
                                data: i,
                                props: o,
                                accessCache: r,
                                type: l,
                                appContext: s
                            } = t;
                            if ("__v_skip" === e) return !0;
                            let c;
                            if ("$" !== e[0]) {
                                const l = r[e];
                                if (void 0 !== l) switch (l) {
                                    case 0:
                                        return a[e];
                                    case 1:
                                        return i[e];
                                    case 3:
                                        return n[e];
                                    case 2:
                                        return o[e]
                                } else {
                                    if (a !== ue && be(a, e)) return r[e] = 0, a[e];
                                    if (i !== ue && be(i, e)) return r[e] = 1, i[e];
                                    if ((c = t.propsOptions[0]) && be(c, e)) return r[e] = 2, o[e];
                                    if (n !== ue && be(n, e)) return r[e] = 3, n[e];
                                    __VUE_OPTIONS_API__ && Yi || (r[e] = 4)
                                }
                            }
                            const u = so[e];
                            let d, p;
                            return u ? ("$attrs" === e && ln(t, 0, e), u(t)) : (d = l.__cssModules) && (d = d[e]) ? d : n !== ue && be(n, e) ? (r[e] = 3, n[e]) : (p = s.config.globalProperties, be(p, e) ? p[e] : void 0)
                        },
                        set({
                            _: t
                        }, e, n) {
                            const {
                                data: a,
                                setupState: i,
                                ctx: o
                            } = t;
                            if (i !== ue && be(i, e)) i[e] = n;
                            else if (a !== ue && be(a, e)) a[e] = n;
                            else if (e in t.props) return !1;
                            return !("$" === e[0] && e.slice(1) in t || (o[e] = n, 0))
                        },
                        has({
                            _: {
                                data: t,
                                setupState: e,
                                accessCache: n,
                                ctx: a,
                                appContext: i,
                                propsOptions: o
                            }
                        }, r) {
                            let l;
                            return void 0 !== n[r] || t !== ue && be(t, r) || e !== ue && be(e, r) || (l = o[0]) && be(l, r) || be(a, r) || be(so, r) || be(i.config.globalProperties, r)
                        }
                    },
                    uo = Ce({}, co, {
                        get(t, e) {
                            if (e !== Symbol.unscopables) return co.get(t, e, t)
                        },
                        has: (t, e) => "_" !== e[0] && !ee(e)
                    }),
                    po = yi();
                let fo = 0,
                    go = null;
                const mo = t => {
                    go = t
                };
                let ho = !1;

                function Co(t, e, n) {
                    xe(e) ? t.render = e : Ne(e) && (__VUE_PROD_DEVTOOLS__ && (t.devtoolsRawSetupState = e), t.setupState = ea(e)), yo(t)
                }

                function yo(t, e) {
                    const n = t.type;
                    t.render || (t.render = n.render || pe, t.render._rc && (t.withProxy = new Proxy(t.ctx, uo))), __VUE_OPTIONS_API__ && (go = t, on(), Qi(t, n), rn(), go = null)
                }

                function vo(t, e = go) {
                    e && (e.effects || (e.effects = [])).push(t)
                }

                function bo(t) {
                    const e = function(t) {
                        let e, n;
                        return xe(t) ? (e = t, n = pe) : (e = t.get, n = t.set), new aa(e, n, xe(t) || !t.set)
                    }(t);
                    return vo(e.effect), e
                }

                function _o(t, e) {
                    let n;
                    if (_e(t) || ke(t)) {
                        n = new Array(t.length);
                        for (let a = 0, i = t.length; a < i; a++) n[a] = e(t[a], a)
                    } else if ("number" == typeof t) {
                        n = new Array(t);
                        for (let a = 0; a < t; a++) n[a] = e(a + 1, a)
                    } else if (Ne(t))
                        if (t[Symbol.iterator]) n = Array.from(t, e);
                        else {
                            const a = Object.keys(t);
                            n = new Array(a.length);
                            for (let i = 0, o = a.length; i < o; i++) {
                                const o = a[i];
                                n[i] = e(t[o], o, i)
                            }
                        }
                    else n = [];
                    return n
                }
                const Lo = "3.0.4",
                    wo = "http://www.w3.org/2000/svg",
                    xo = "undefined" != typeof document ? document : null;
                let ko, So;
                const No = {
                        insert: (t, e, n) => {
                            e.insertBefore(t, n || null)
                        },
                        remove: t => {
                            const e = t.parentNode;
                            e && e.removeChild(t)
                        },
                        createElement: (t, e, n) => e ? xo.createElementNS(wo, t) : xo.createElement(t, n ? {
                            is: n
                        } : void 0),
                        createText: t => xo.createTextNode(t),
                        createComment: t => xo.createComment(t),
                        setText: (t, e) => {
                            t.nodeValue = e
                        },
                        setElementText: (t, e) => {
                            t.textContent = e
                        },
                        parentNode: t => t.parentNode,
                        nextSibling: t => t.nextSibling,
                        querySelector: t => xo.querySelector(t),
                        setScopeId(t, e) {
                            t.setAttribute(e, "")
                        },
                        cloneNode: t => t.cloneNode(!0),
                        insertStaticContent(t, e, n, a) {
                            const i = a ? So || (So = xo.createElementNS(wo, "svg")) : ko || (ko = xo.createElement("div"));
                            i.innerHTML = t;
                            const o = i.firstChild;
                            let r = o,
                                l = r;
                            for (; r;) l = r, No.insert(r, e, n), r = i.firstChild;
                            return [o, l]
                        }
                    },
                    Mo = /\s*!important$/;

                function To(t, e, n) {
                    if (_e(n)) n.forEach((n => To(t, e, n)));
                    else if (e.startsWith("--")) t.setProperty(e, n);
                    else {
                        const a = function(t, e) {
                            const n = Do[e];
                            if (n) return n;
                            let a = Re(e);
                            if ("filter" !== a && a in t) return Do[e] = a;
                            a = je(a);
                            for (let n = 0; n < Oo.length; n++) {
                                const i = Oo[n] + a;
                                if (i in t) return Do[e] = i
                            }
                            return e
                        }(t, e);
                        Mo.test(n) ? t.setProperty(Be(a), n.replace(Mo, ""), "important") : t[a] = n
                    }
                }
                const Oo = ["Webkit", "Moz", "ms"],
                    Do = {},
                    Ao = "http://www.w3.org/1999/xlink";
                let Eo = Date.now;
                "undefined" != typeof document && Eo() > document.createEvent("Event").timeStamp && (Eo = () => performance.now());
                let Po = 0;
                const Fo = Promise.resolve(),
                    Ro = () => {
                        Po = 0
                    },
                    Io = /(?:Once|Passive|Capture)$/,
                    Bo = /^on[a-z]/,
                    jo = {
                        beforeMount(t, {
                            value: e
                        }, {
                            transition: n
                        }) {
                            t._vod = "none" === t.style.display ? "" : t.style.display, n && e ? n.beforeEnter(t) : Go(t, e)
                        },
                        mounted(t, {
                            value: e
                        }, {
                            transition: n
                        }) {
                            n && e && n.enter(t)
                        },
                        updated(t, {
                            value: e,
                            oldValue: n
                        }, {
                            transition: a
                        }) {
                            a && e !== n ? e ? (a.beforeEnter(t), Go(t, !0), a.enter(t)) : a.leave(t, (() => {
                                Go(t, !1)
                            })) : Go(t, e)
                        },
                        beforeUnmount(t, {
                            value: e
                        }) {
                            Go(t, e)
                        }
                    };

                function Go(t, e) {
                    t.style.display = e ? t._vod : "none"
                }
                const zo = Ce({
                    patchProp: (t, e, n, a, i = !1, o, r, l, s) => {
                        switch (e) {
                            case "class":
                                ! function(t, e, n) {
                                    if (null == e && (e = ""), n) t.setAttribute("class", e);
                                    else {
                                        const n = t._vtc;
                                        n && (e = (e ? [e, ...n] : [...n]).join(" ")), t.className = e
                                    }
                                }(t, a, i);
                                break;
                            case "style":
                                ! function(t, e, n) {
                                    const a = t.style;
                                    if (n)
                                        if (ke(n)) e !== n && (a.cssText = n);
                                        else {
                                            for (const t in n) To(a, t, n[t]);
                                            if (e && !ke(e))
                                                for (const t in e) null == n[t] && To(a, t, "")
                                        }
                                    else t.removeAttribute("style")
                                }(t, n, a);
                                break;
                            default:
                                me(e) ? he(e) || function(t, e, n, a, i = null) {
                                    const o = t._vei || (t._vei = {}),
                                        r = o[e];
                                    if (a && r) r.value = a;
                                    else {
                                        const [n, l] = function(t) {
                                            let e;
                                            if (Io.test(t)) {
                                                let n;
                                                for (e = {}; n = t.match(Io);) t = t.slice(0, t.length - n[0].length), e[n[0].toLowerCase()] = !0
                                            }
                                            return [t.slice(2).toLowerCase(), e]
                                        }(e);
                                        a ? function(t, e, n, a) {
                                            t.addEventListener(e, n, a)
                                        }(t, n, o[e] = function(t, e) {
                                            const n = t => {
                                                (t.timeStamp || Eo()) >= n.attached - 1 && oa(function(t, e) {
                                                    if (_e(e)) {
                                                        const n = t.stopImmediatePropagation;
                                                        return t.stopImmediatePropagation = () => {
                                                            n.call(t), t._stopped = !0
                                                        }, e.map((t => e => !e._stopped && t(e)))
                                                    }
                                                    return e
                                                }(t, n.value), e, 5, [t])
                                            };
                                            return n.value = t, n.attached = Po || (Fo.then(Ro), Po = Eo()), n
                                        }(a, i), l) : r && (function(t, e, n, a) {
                                            t.removeEventListener(e, n, a)
                                        }(t, n, r, l), o[e] = void 0)
                                    }
                                }(t, e, 0, a, r) : function(t, e, n, a) {
                                    return a ? "innerHTML" === e || !!(e in t && Bo.test(e) && xe(n)) : !("spellcheck" === e || "draggable" === e || "form" === e && "string" == typeof n || "list" === e && "INPUT" === t.tagName || Bo.test(e) && ke(n) || !(e in t))
                                }(t, e, a, i) ? function(t, e, n, a, i, o, r) {
                                    if ("innerHTML" === e || "textContent" === e) return a && r(a, i, o), void(t[e] = null == n ? "" : n);
                                    if ("value" !== e || "PROGRESS" === t.tagName) {
                                        if ("" === n || null == n) {
                                            const a = typeof t[e];
                                            if ("" === n && "boolean" === a) return void(t[e] = !0);
                                            if (null == n && "string" === a) return t[e] = "", void t.removeAttribute(e);
                                            if ("number" === a) return t[e] = 0, void t.removeAttribute(e)
                                        }
                                        try {
                                            t[e] = n
                                        } catch (t) {}
                                    } else {
                                        t._value = n;
                                        const e = null == n ? "" : n;
                                        t.value !== e && (t.value = e)
                                    }
                                }(t, e, a, o, r, l, s) : ("true-value" === e ? t._trueValue = a : "false-value" === e && (t._falseValue = a), function(t, e, n, a) {
                                    if (a && e.startsWith("xlink:")) null == n ? t.removeAttributeNS(Ao, e.slice(6, e.length)) : t.setAttributeNS(Ao, e, n);
                                    else {
                                        const a = ne(e);
                                        null == n || a && !1 === n ? t.removeAttribute(e) : t.setAttribute(e, a ? "" : n)
                                    }
                                }(t, e, a, i))
                        }
                    },
                    forcePatchProp: (t, e) => "value" === e
                }, No);
                let $o;

                function Uo() {
                    return $o || ($o = function(t) {
                        return function(t, e) {
                            "boolean" != typeof __VUE_OPTIONS_API__ && (We().__VUE_OPTIONS_API__ = !0), "boolean" != typeof __VUE_PROD_DEVTOOLS__ && (We().__VUE_PROD_DEVTOOLS__ = !1);
                            const {
                                insert: n,
                                remove: a,
                                patchProp: i,
                                forcePatchProp: o,
                                createElement: r,
                                createText: l,
                                createComment: s,
                                setText: c,
                                setElementText: u,
                                parentNode: d,
                                nextSibling: p,
                                setScopeId: f = pe,
                                cloneNode: g,
                                insertStaticContent: m
                            } = t, h = (t, e, n, a = null, i = null, o = null, r = !1, l = !1) => {
                                t && !Gi(t, e) && (a = $(t), I(t, i, o, !0), t = null), -2 === e.patchFlag && (l = !1, e.dynamicChildren = null);
                                const {
                                    type: s,
                                    ref: c,
                                    shapeFlag: u
                                } = e;
                                switch (s) {
                                    case Di:
                                        C(t, e, n, a);
                                        break;
                                    case Ai:
                                        y(t, e, n, a);
                                        break;
                                    case Ei:
                                        null == t && v(e, n, a, r);
                                        break;
                                    case Oi:
                                        N(t, e, n, a, i, o, r, l);
                                        break;
                                    default:
                                        1 & u ? b(t, e, n, a, i, o, r, l) : 6 & u ? M(t, e, n, a, i, o, r, l) : (64 & u || 128 & u) && s.process(t, e, n, a, i, o, r, l, Z)
                                }
                                null != c && i && wi(c, t && t.ref, o, e)
                            }, C = (t, e, a, i) => {
                                if (null == t) n(e.el = l(e.children), a, i);
                                else {
                                    const n = e.el = t.el;
                                    e.children !== t.children && c(n, e.children)
                                }
                            }, y = (t, e, a, i) => {
                                null == t ? n(e.el = s(e.children || ""), a, i) : e.el = t.el
                            }, v = (t, e, n, a) => {
                                [t.el, t.anchor] = m(t.children, e, n, a)
                            }, b = (t, e, n, a, i, o, r, l) => {
                                r = r || "svg" === e.type, null == t ? _(e, n, a, i, o, r, l) : x(t, e, i, o, r, l)
                            }, _ = (t, e, a, o, l, s, c) => {
                                let d, p;
                                const {
                                    type: f,
                                    props: m,
                                    shapeFlag: h,
                                    transition: C,
                                    scopeId: y,
                                    patchFlag: v,
                                    dirs: b
                                } = t;
                                if (t.el && void 0 !== g && -1 === v) d = t.el = g(t.el);
                                else {
                                    if (d = t.el = r(t.type, s, m && m.is), 8 & h ? u(d, t.children) : 16 & h && w(t.children, d, null, o, l, s && "foreignObject" !== f, c || !!t.dynamicChildren), b && Ci(t, null, o, "created"), m) {
                                        for (const e in m) Ee(e) || i(d, e, null, m[e], s, t.children, o, l, z);
                                        (p = m.onVnodeBeforeMount) && xi(p, o, t)
                                    }
                                    L(d, y, t, o)
                                }
                                __VUE_PROD_DEVTOOLS__ && (Object.defineProperty(d, "__vnode", {
                                    value: t,
                                    enumerable: !1
                                }), Object.defineProperty(d, "__vueParentComponent", {
                                    value: o,
                                    enumerable: !1
                                })), b && Ci(t, null, o, "beforeMount");
                                const _ = (!l || l && !l.pendingBranch) && C && !C.persisted;
                                _ && C.beforeEnter(d), n(d, e, a), ((p = m && m.onVnodeMounted) || _ || b) && Li((() => {
                                    p && xi(p, o, t), _ && C.enter(d), b && Ci(t, null, o, "mounted")
                                }), l)
                            }, L = (t, e, n, a) => {
                                if (e && f(t, e), a) {
                                    const i = a.type.__scopeId;
                                    i && i !== e && f(t, i + "-s"), n === a.subTree && L(t, a.vnode.scopeId, a.vnode, a.parent)
                                }
                            }, w = (t, e, n, a, i, o, r, l = 0) => {
                                for (let s = l; s < t.length; s++) {
                                    const l = t[s] = r ? Ki(t[s]) : qi(t[s]);
                                    h(null, l, e, n, a, i, o, r)
                                }
                            }, x = (t, e, n, a, r, l) => {
                                const s = e.el = t.el;
                                let {
                                    patchFlag: c,
                                    dynamicChildren: d,
                                    dirs: p
                                } = e;
                                c |= 16 & t.patchFlag;
                                const f = t.props || ue,
                                    g = e.props || ue;
                                let m;
                                if ((m = g.onVnodeBeforeUpdate) && xi(m, n, e, t), p && Ci(e, t, n, "beforeUpdate"), c > 0) {
                                    if (16 & c) S(s, e, f, g, n, a, r);
                                    else if (2 & c && f.class !== g.class && i(s, "class", null, g.class, r), 4 & c && i(s, "style", f.style, g.style, r), 8 & c) {
                                        const l = e.dynamicProps;
                                        for (let e = 0; e < l.length; e++) {
                                            const c = l[e],
                                                u = f[c],
                                                d = g[c];
                                            (d !== u || o && o(s, c)) && i(s, c, u, d, r, t.children, n, a, z)
                                        }
                                    }
                                    1 & c && t.children !== e.children && u(s, e.children)
                                } else l || null != d || S(s, e, f, g, n, a, r);
                                const h = r && "foreignObject" !== e.type;
                                d ? k(t.dynamicChildren, d, s, n, a, h) : l || E(t, e, s, null, n, a, h), ((m = g.onVnodeUpdated) || p) && Li((() => {
                                    m && xi(m, n, e, t), p && Ci(e, t, n, "updated")
                                }), a)
                            }, k = (t, e, n, a, i, o) => {
                                for (let r = 0; r < e.length; r++) {
                                    const l = t[r],
                                        s = e[r],
                                        c = l.type === Oi || !Gi(l, s) || 6 & l.shapeFlag || 64 & l.shapeFlag ? d(l.el) : n;
                                    h(l, s, c, null, a, i, o, !0)
                                }
                            }, S = (t, e, n, a, r, l, s) => {
                                if (n !== a) {
                                    for (const c in a) {
                                        if (Ee(c)) continue;
                                        const u = a[c],
                                            d = n[c];
                                        (u !== d || o && o(t, c)) && i(t, c, d, u, s, e.children, r, l, z)
                                    }
                                    if (n !== ue)
                                        for (const o in n) Ee(o) || o in a || i(t, o, n[o], null, s, e.children, r, l, z)
                                }
                            }, N = (t, e, a, i, o, r, s, c) => {
                                const u = e.el = t ? t.el : l(""),
                                    d = e.anchor = t ? t.anchor : l("");
                                let {
                                    patchFlag: p,
                                    dynamicChildren: f
                                } = e;
                                p > 0 && (c = !0), null == t ? (n(u, a, i), n(d, a, i), w(e.children, a, d, o, r, s, c)) : p > 0 && 64 & p && f ? (k(t.dynamicChildren, f, a, o, r, s), (null != e.key || o && e === o.subTree) && ki(t, e, !0)) : E(t, e, a, d, o, r, s, c)
                            }, M = (t, e, n, a, i, o, r, l) => {
                                null == t ? 512 & e.shapeFlag ? i.ctx.activate(e, n, a, r, l) : T(e, n, a, i, o, r, l) : O(t, e, l)
                            }, T = (t, e, n, a, i, o, r) => {
                                const l = t.component = function(t, e, n) {
                                    const a = t.type,
                                        i = (e ? e.appContext : t.appContext) || po,
                                        o = {
                                            uid: fo++,
                                            vnode: t,
                                            type: a,
                                            parent: e,
                                            appContext: i,
                                            root: null,
                                            next: null,
                                            subTree: null,
                                            update: null,
                                            render: null,
                                            proxy: null,
                                            exposed: null,
                                            withProxy: null,
                                            effects: null,
                                            provides: e ? e.provides : Object.create(i.provides),
                                            accessCache: null,
                                            renderCache: [],
                                            components: null,
                                            directives: null,
                                            propsOptions: $a(a, i),
                                            emitsOptions: Ta(a, i),
                                            emit: null,
                                            emitted: null,
                                            ctx: ue,
                                            data: ue,
                                            props: ue,
                                            attrs: ue,
                                            slots: ue,
                                            refs: ue,
                                            setupState: ue,
                                            setupContext: null,
                                            suspense: n,
                                            suspenseId: n ? n.pendingId : 0,
                                            asyncDep: null,
                                            asyncResolved: !1,
                                            isMounted: !1,
                                            isUnmounted: !1,
                                            isDeactivated: !1,
                                            bc: null,
                                            c: null,
                                            bm: null,
                                            m: null,
                                            bu: null,
                                            u: null,
                                            um: null,
                                            bum: null,
                                            da: null,
                                            a: null,
                                            rtg: null,
                                            rtc: null,
                                            ec: null
                                        };
                                    return o.ctx = {
                                        _: o
                                    }, o.root = e ? e.root : o, o.emit = Ma.bind(null, o), __VUE_PROD_DEVTOOLS__, o
                                }(t, a, i);
                                if (si(t) && (l.ctx.renderer = Z), function(t, e = !1) {
                                        ho = e;
                                        const {
                                            props: n,
                                            children: a,
                                            shapeFlag: i
                                        } = t.vnode, o = 4 & i;
                                        (function(t, e, n, a = !1) {
                                            const i = {},
                                                o = {};
                                            Ue(o, zi, 1), Ga(t, e, i, o), n ? t.props = a ? i : Vn(i, !1, vn, jn) : t.type.props ? t.props = i : t.props = o, t.attrs = o
                                        })(t, n, o, e), ((t, e) => {
                                            if (32 & t.vnode.shapeFlag) {
                                                const n = e._;
                                                n ? (t.slots = e, Ue(e, "_", n)) : gi(e, t.slots = {})
                                            } else t.slots = {}, e && mi(t, e);
                                            Ue(t.slots, zi, 1)
                                        })(t, a), o && function(t, e) {
                                            const n = t.type;
                                            t.accessCache = Object.create(null), t.proxy = new Proxy(t.ctx, co);
                                            const {
                                                setup: a
                                            } = n;
                                            if (a) {
                                                const n = t.setupContext = a.length > 1 ? function(t) {
                                                    return {
                                                        attrs: t.attrs,
                                                        slots: t.slots,
                                                        emit: t.emit,
                                                        expose: e => {
                                                            t.exposed = ea(e)
                                                        }
                                                    }
                                                }(t) : null;
                                                go = t, on();
                                                const i = ia(a, t, 0, [t.props, n]);
                                                if (rn(), go = null, Me(i)) {
                                                    if (e) return i.then((e => {
                                                        Co(t, e)
                                                    }));
                                                    t.asyncDep = i
                                                } else Co(t, i)
                                            } else yo(t)
                                        }(t, e), ho = !1
                                    }(l), l.asyncDep) {
                                    if (i && i.registerDep(l, D), !t.el) {
                                        const t = l.subTree = Zi(Ai);
                                        y(null, t, e, n)
                                    }
                                } else D(l, t, e, n, i, o, r)
                            }, O = (t, e, n) => {
                                const a = e.component = t.component;
                                if (function(t, e, n) {
                                        const {
                                            props: a,
                                            children: i,
                                            component: o
                                        } = t, {
                                            props: r,
                                            children: l,
                                            patchFlag: s
                                        } = e, c = o.emitsOptions;
                                        if (e.dirs || e.transition) return !0;
                                        if (!(n && s >= 0)) return !(!i && !l || l && l.$stable) || a !== r && (a ? !r || Ra(a, r, c) : !!r);
                                        if (1024 & s) return !0;
                                        if (16 & s) return a ? Ra(a, r, c) : !!r;
                                        if (8 & s) {
                                            const t = e.dynamicProps;
                                            for (let e = 0; e < t.length; e++) {
                                                const n = t[e];
                                                if (r[n] !== a[n] && !Oa(c, n)) return !0
                                            }
                                        }
                                        return !1
                                    }(t, e, n)) {
                                    if (a.asyncDep && !a.asyncResolved) return void A(a, e, n);
                                    a.next = e,
                                        function(t) {
                                            const e = ca.indexOf(t);
                                            e > -1 && ca.splice(e, 1)
                                        }(a.update), a.update()
                                } else e.component = t.component, e.el = t.el, a.vnode = e
                            }, D = (t, e, n, a, i, o, r) => {
                                t.update = Ye((function() {
                                    if (t.isMounted) {
                                        let e, {
                                                next: n,
                                                bu: a,
                                                u: l,
                                                parent: s,
                                                vnode: c
                                            } = t,
                                            u = n;
                                        n ? (n.el = c.el, A(t, n, r)) : n = c, a && $e(a), (e = n.props && n.props.onVnodeBeforeUpdate) && xi(e, s, n, c);
                                        const p = Ea(t),
                                            f = t.subTree;
                                        t.subTree = p, h(f, p, d(f.el), $(f), t, i, o), n.el = p.el, null === u && function({
                                            vnode: t,
                                            parent: e
                                        }, n) {
                                            for (; e && e.subTree === t;)(t = e.vnode).el = n, e = e.parent
                                        }(t, p.el), l && Li(l, i), (e = n.props && n.props.onVnodeUpdated) && Li((() => {
                                            xi(e, s, n, c)
                                        }), i), __VUE_PROD_DEVTOOLS__
                                    } else {
                                        let r;
                                        const {
                                            el: l,
                                            props: s
                                        } = e, {
                                            bm: c,
                                            m: u,
                                            parent: d
                                        } = t;
                                        c && $e(c), (r = s && s.onVnodeBeforeMount) && xi(r, d, e);
                                        const p = t.subTree = Ea(t);
                                        h(null, p, n, a, t, i, o), e.el = p.el, u && Li(u, i), (r = s && s.onVnodeMounted) && Li((() => {
                                            xi(r, d, e)
                                        }), i);
                                        const {
                                            a: f
                                        } = t;
                                        f && 256 & e.shapeFlag && Li(f, i), t.isMounted = !0
                                    }
                                }), _i)
                            }, A = (t, e, n) => {
                                e.component = t;
                                const a = t.vnode.props;
                                t.vnode = e, t.next = null,
                                    function(t, e, n, a) {
                                        const {
                                            props: i,
                                            attrs: o,
                                            vnode: {
                                                patchFlag: r
                                            }
                                        } = t, l = Kn(i), [s] = t.propsOptions;
                                        if (!(a || r > 0) || 16 & r) {
                                            let a;
                                            Ga(t, e, i, o);
                                            for (const o in l) e && (be(e, o) || (a = Be(o)) !== o && be(e, a)) || (s ? !n || void 0 === n[o] && void 0 === n[a] || (i[o] = za(s, e || ue, o, void 0, t)) : delete i[o]);
                                            if (o !== l)
                                                for (const t in o) e && be(e, t) || delete o[t]
                                        } else if (8 & r) {
                                            const n = t.vnode.dynamicProps;
                                            for (let a = 0; a < n.length; a++) {
                                                const r = n[a],
                                                    c = e[r];
                                                if (s)
                                                    if (be(o, r)) o[r] = c;
                                                    else {
                                                        const e = Re(r);
                                                        i[e] = za(s, l, e, c, t)
                                                    }
                                                else o[r] = c
                                            }
                                        }
                                        sn(t, "set", "$attrs")
                                    }(t, e.props, a, n), ((t, e) => {
                                        const {
                                            vnode: n,
                                            slots: a
                                        } = t;
                                        let i = !0,
                                            o = ue;
                                        if (32 & n.shapeFlag) {
                                            const t = e._;
                                            t ? 1 === t ? i = !1 : Ce(a, e) : (i = !e.$stable, gi(e, a)), o = e
                                        } else e && (mi(t, e), o = {
                                            default: 1
                                        });
                                        if (i)
                                            for (const t in a) di(t) || t in o || delete a[t]
                                    })(t, e.children), xa(void 0, t.update)
                            }, E = (t, e, n, a, i, o, r, l = !1) => {
                                const s = t && t.children,
                                    c = t ? t.shapeFlag : 0,
                                    d = e.children,
                                    {
                                        patchFlag: p,
                                        shapeFlag: f
                                    } = e;
                                if (p > 0) {
                                    if (128 & p) return void F(s, d, n, a, i, o, r, l);
                                    if (256 & p) return void P(s, d, n, a, i, o, r, l)
                                }
                                8 & f ? (16 & c && z(s, i, o), d !== s && u(n, d)) : 16 & c ? 16 & f ? F(s, d, n, a, i, o, r, l) : z(s, i, o, !0) : (8 & c && u(n, ""), 16 & f && w(d, n, a, i, o, r, l))
                            }, P = (t, e, n, a, i, o, r, l) => {
                                e = e || de;
                                const s = (t = t || de).length,
                                    c = e.length,
                                    u = Math.min(s, c);
                                let d;
                                for (d = 0; d < u; d++) {
                                    const a = e[d] = l ? Ki(e[d]) : qi(e[d]);
                                    h(t[d], a, n, null, i, o, r, l)
                                }
                                s > c ? z(t, i, o, !0, !1, u) : w(e, n, a, i, o, r, l, u)
                            }, F = (t, e, n, a, i, o, r, l) => {
                                let s = 0;
                                const c = e.length;
                                let u = t.length - 1,
                                    d = c - 1;
                                for (; s <= u && s <= d;) {
                                    const a = t[s],
                                        c = e[s] = l ? Ki(e[s]) : qi(e[s]);
                                    if (!Gi(a, c)) break;
                                    h(a, c, n, null, i, o, r, l), s++
                                }
                                for (; s <= u && s <= d;) {
                                    const a = t[u],
                                        s = e[d] = l ? Ki(e[d]) : qi(e[d]);
                                    if (!Gi(a, s)) break;
                                    h(a, s, n, null, i, o, r, l), u--, d--
                                }
                                if (s > u) {
                                    if (s <= d) {
                                        const t = d + 1,
                                            u = t < c ? e[t].el : a;
                                        for (; s <= d;) h(null, e[s] = l ? Ki(e[s]) : qi(e[s]), n, u, i, o, r), s++
                                    }
                                } else if (s > d)
                                    for (; s <= u;) I(t[s], i, o, !0), s++;
                                else {
                                    const p = s,
                                        f = s,
                                        g = new Map;
                                    for (s = f; s <= d; s++) {
                                        const t = e[s] = l ? Ki(e[s]) : qi(e[s]);
                                        null != t.key && g.set(t.key, s)
                                    }
                                    let m, C = 0;
                                    const y = d - f + 1;
                                    let v = !1,
                                        b = 0;
                                    const _ = new Array(y);
                                    for (s = 0; s < y; s++) _[s] = 0;
                                    for (s = p; s <= u; s++) {
                                        const a = t[s];
                                        if (C >= y) {
                                            I(a, i, o, !0);
                                            continue
                                        }
                                        let c;
                                        if (null != a.key) c = g.get(a.key);
                                        else
                                            for (m = f; m <= d; m++)
                                                if (0 === _[m - f] && Gi(a, e[m])) {
                                                    c = m;
                                                    break
                                                }
                                        void 0 === c ? I(a, i, o, !0) : (_[c - f] = s + 1, c >= b ? b = c : v = !0, h(a, e[c], n, null, i, o, r, l), C++)
                                    }
                                    const L = v ? function(t) {
                                        const e = t.slice(),
                                            n = [0];
                                        let a, i, o, r, l;
                                        const s = t.length;
                                        for (a = 0; a < s; a++) {
                                            const s = t[a];
                                            if (0 !== s) {
                                                if (i = n[n.length - 1], t[i] < s) {
                                                    e[a] = i, n.push(a);
                                                    continue
                                                }
                                                for (o = 0, r = n.length - 1; o < r;) l = (o + r) / 2 | 0, t[n[l]] < s ? o = l + 1 : r = l;
                                                s < t[n[o]] && (o > 0 && (e[a] = n[o - 1]), n[o] = a)
                                            }
                                        }
                                        for (o = n.length, r = n[o - 1]; o-- > 0;) n[o] = r, r = e[r];
                                        return n
                                    }(_) : de;
                                    for (m = L.length - 1, s = y - 1; s >= 0; s--) {
                                        const t = f + s,
                                            l = e[t],
                                            u = t + 1 < c ? e[t + 1].el : a;
                                        0 === _[s] ? h(null, l, n, u, i, o, r) : v && (m < 0 || s !== L[m] ? R(l, n, u, 2) : m--)
                                    }
                                }
                            }, R = (t, e, a, i, o = null) => {
                                const {
                                    el: r,
                                    type: l,
                                    transition: s,
                                    children: c,
                                    shapeFlag: u
                                } = t;
                                if (6 & u) R(t.component.subTree, e, a, i);
                                else if (128 & u) t.suspense.move(e, a, i);
                                else if (64 & u) l.move(t, e, a, Z);
                                else if (l !== Oi)
                                    if (l !== Ei)
                                        if (2 !== i && 1 & u && s)
                                            if (0 === i) s.beforeEnter(r), n(r, e, a), Li((() => s.enter(r)), o);
                                            else {
                                                const {
                                                    leave: t,
                                                    delayLeave: i,
                                                    afterLeave: o
                                                } = s, l = () => n(r, e, a), c = () => {
                                                    t(r, (() => {
                                                        l(), o && o()
                                                    }))
                                                };
                                                i ? i(r, l, c) : c()
                                            }
                                else n(r, e, a);
                                else(({
                                    el: t,
                                    anchor: e
                                }, a, i) => {
                                    let o;
                                    for (; t && t !== e;) o = p(t), n(t, a, i), t = o;
                                    n(e, a, i)
                                })(t, e, a);
                                else {
                                    n(r, e, a);
                                    for (let t = 0; t < c.length; t++) R(c[t], e, a, i);
                                    n(t.anchor, e, a)
                                }
                            }, I = (t, e, n, a = !1, i = !1) => {
                                const {
                                    type: o,
                                    props: r,
                                    ref: l,
                                    children: s,
                                    dynamicChildren: c,
                                    shapeFlag: u,
                                    patchFlag: d,
                                    dirs: p
                                } = t;
                                if (null != l && wi(l, null, n, null), 256 & u) return void e.ctx.deactivate(t);
                                const f = 1 & u && p;
                                let g;
                                if ((g = r && r.onVnodeBeforeUnmount) && xi(g, e, t), 6 & u) G(t.component, n, a);
                                else {
                                    if (128 & u) return void t.suspense.unmount(n, a);
                                    f && Ci(t, null, e, "beforeUnmount"), c && (o !== Oi || d > 0 && 64 & d) ? z(c, e, n, !1, !0) : (o === Oi && (128 & d || 256 & d) || !i && 16 & u) && z(s, e, n), 64 & u && (a || !Si(t.props)) && t.type.remove(t, Z), a && B(t)
                                }((g = r && r.onVnodeUnmounted) || f) && Li((() => {
                                    g && xi(g, e, t), f && Ci(t, null, e, "unmounted")
                                }), n)
                            }, B = t => {
                                const {
                                    type: e,
                                    el: n,
                                    anchor: i,
                                    transition: o
                                } = t;
                                if (e === Oi) return void j(n, i);
                                if (e === Ei) return void(({
                                    el: t,
                                    anchor: e
                                }) => {
                                    let n;
                                    for (; t && t !== e;) n = p(t), a(t), t = n;
                                    a(e)
                                })(t);
                                const r = () => {
                                    a(n), o && !o.persisted && o.afterLeave && o.afterLeave()
                                };
                                if (1 & t.shapeFlag && o && !o.persisted) {
                                    const {
                                        leave: e,
                                        delayLeave: a
                                    } = o, i = () => e(n, r);
                                    a ? a(t.el, r, i) : i()
                                } else r()
                            }, j = (t, e) => {
                                let n;
                                for (; t !== e;) n = p(t), a(t), t = n;
                                a(e)
                            }, G = (t, e, n) => {
                                const {
                                    bum: a,
                                    effects: i,
                                    update: o,
                                    subTree: r,
                                    um: l
                                } = t;
                                if (a && $e(a), i)
                                    for (let t = 0; t < i.length; t++) Qe(i[t]);
                                o && (Qe(o), I(r, t, e, n)), l && Li(l, e), Li((() => {
                                    t.isUnmounted = !0
                                }), e), e && e.pendingBranch && !e.isUnmounted && t.asyncDep && !t.asyncResolved && t.suspenseId === e.pendingId && (e.deps--, 0 === e.deps && e.resolve()), __VUE_PROD_DEVTOOLS__
                            }, z = (t, e, n, a = !1, i = !1, o = 0) => {
                                for (let r = o; r < t.length; r++) I(t[r], e, n, a, i)
                            }, $ = t => 6 & t.shapeFlag ? $(t.component.subTree) : 128 & t.shapeFlag ? t.suspense.next() : p(t.anchor || t.el), U = (t, e) => {
                                null == t ? e._vnode && I(e._vnode, null, null, !0) : h(e._vnode || null, t, e), ka(), e._vnode = t
                            }, Z = {
                                p: h,
                                um: I,
                                m: R,
                                r: B,
                                mt: T,
                                mc: w,
                                pc: E,
                                pbc: k,
                                n: $,
                                o: t
                            };
                            let V;
                            return {
                                render: U,
                                hydrate: V,
                                createApp: bi(U, V)
                            }
                        }(t)
                    }(zo))
                }
                const Zo = (...t) => Object.prototype.toString.call(...t).slice(8, -1);
                var Vo = {
                    props: {
                        data: {
                            required: !0,
                            validator: t => "Null" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        }
                    }
                };
                const Wo = {
                        class: "null"
                    },
                    Ho = {
                        class: "key"
                    },
                    qo = {
                        key: 0,
                        class: "separator"
                    },
                    Ko = Zi("span", {
                        class: "value"
                    }, "null", -1);
                Vo.render = function(t, e, n, a, i, o) {
                    return Ri(), Bi("span", Wo, [Zi("span", Ho, se(n.name), 1), "" !== n.name ? (Ri(), Bi("span", qo, ": ")) : Hi("v-if", !0), Ko])
                }, Vo.__file = "src/components/NullWrapper.vue";
                var Jo = {
                    props: {
                        data: {
                            required: !0,
                            validator: t => "Boolean" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        }
                    }
                };
                const Xo = {
                        class: "boolean"
                    },
                    Yo = {
                        class: "key"
                    },
                    Qo = {
                        key: 0,
                        class: "separator"
                    },
                    tr = {
                        class: "value"
                    };
                Jo.render = function(t, e, n, a, i, o) {
                    return Ri(), Bi("span", Xo, [Zi("span", Yo, se(n.name), 1), "" !== n.name ? (Ri(), Bi("span", Qo, ": ")) : Hi("v-if", !0), Zi("span", tr, se(n.data), 1)])
                }, Jo.__file = "src/components/BooleanWrapper.vue";
                var er = {
                    props: {
                        data: {
                            required: !0,
                            validator: t => "Number" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        }
                    }
                };
                const nr = {
                        class: "number"
                    },
                    ar = {
                        class: "key"
                    },
                    ir = {
                        key: 0,
                        class: "separator"
                    },
                    or = {
                        class: "value"
                    };
                er.render = function(t, e, n, a, i, o) {
                    return Ri(), Bi("span", nr, [Zi("span", ar, se(n.name), 1), "" !== n.name ? (Ri(), Bi("span", ir, ": ")) : Hi("v-if", !0), Zi("span", or, se(n.data), 1)])
                }, er.__file = "src/components/NumberWrapper.vue";
                var rr = {
                    props: {
                        data: {
                            required: !0,
                            validator: t => "String" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        }
                    }
                };
                const lr = {
                        class: "string"
                    },
                    sr = {
                        class: "key"
                    },
                    cr = {
                        key: 0,
                        class: "separator"
                    },
                    ur = Zi("span", {
                        class: "quotes"
                    }, '"', -1),
                    dr = {
                        class: "value"
                    },
                    pr = Zi("span", {
                        class: "quotes"
                    }, '"', -1);
                rr.render = function(t, e, n, a, i, o) {
                    return Ri(), Bi("span", lr, [Zi("span", sr, se(n.name), 1), "" !== n.name ? (Ri(), Bi("span", cr, ": ")) : Hi("v-if", !0), ur, Zi("span", dr, se(n.data), 1), pr])
                }, rr.__file = "src/components/StringWrapper.vue";
                const fr = new Set;

                function gr(t = {
                    collapseSignal,
                    expandSignal
                }) {
                    const e = Yn(!1),
                        n = Yn(!1),
                        a = () => {
                            e.value = !1, n.value = !n.value
                        };
                    ii((() => t.collapseSignal), a);
                    const i = Yn(!1),
                        o = () => {
                            e.value = !0, i.value = !i.value
                        };
                    return ii((() => t.expandSignal), o), ii((() => t.data), (() => {
                        t.expandOnCreatedAndUpdated(t.path) ? o() : a()
                    }), {
                        immediate: !0
                    }), {
                        isExpanding: e,
                        innerCollapseSignal: n,
                        innerExpandSignal: i,
                        handleClick: t => {
                            fr.clear(), !0 === t.metaKey && !0 === t.shiftKey ? a() : !0 === t.metaKey ? o() : e.value = !e.value
                        }
                    }
                }
                var mr = {
                    name: "array-wrapper",
                    props: {
                        path: {
                            required: !0,
                            validator: t => "Array" === Zo(t) && t.every((t => "String" === Zo(t) || "Number" === Zo(t)))
                        },
                        data: {
                            required: !0,
                            validator: t => "Array" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        },
                        collapseSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandOnCreatedAndUpdated: {
                            required: !0,
                            type: Function
                        },
                        getKeys: {
                            required: !0,
                            type: Function
                        }
                    },
                    setup(t) {
                        const {
                            isExpanding: e,
                            innerExpandSignal: n,
                            innerCollapseSignal: a,
                            handleClick: i
                        } = gr(t), o = bo((() => t.getKeys(t.data, t.path))), r = fr.has(t.data);
                        return fr.add(t.data), {
                            keys: o,
                            isExpanding: e,
                            innerExpandSignal: n,
                            innerCollapseSignal: a,
                            handleClick: i,
                            isCircular: r
                        }
                    },
                    components: {}
                };
                const hr = {
                        class: "array"
                    },
                    Cr = {
                        key: 0,
                        class: "value"
                    },
                    yr = {
                        key: 0,
                        class: "value"
                    };
                mr.render = function(t, e, n, a, i, o) {
                    const r = Ni("wrapper");
                    return Ri(), Bi("span", hr, [Zi("span", {
                        class: "indicator",
                        onClick: e[1] || (e[1] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se(a.isExpanding ? "▼" : "▶"), 1), Zi("span", {
                        class: "key",
                        onClick: e[2] || (e[2] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se("" === n.name ? "" : n.name), 1), Zi("span", {
                        class: "separator",
                        onClick: e[3] || (e[3] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se("" === n.name ? "" : ": "), 1), Zi("span", {
                        class: "count",
                        onClick: e[4] || (e[4] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se(!1 === a.isExpanding && n.data.length >= 2 ? "(" + n.data.length + ")" : ""), 1), Zi("span", {
                        class: "preview",
                        onClick: e[5] || (e[5] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se(a.isExpanding ? "Array(" + n.data.length + ")" : "[...]"), 1), a.isCircular ? (Ri(), Bi(Oi, {
                        key: 0
                    }, [a.isExpanding ? (Ri(), Bi("span", Cr, [(Ri(!0), Bi(Oi, null, _o(a.keys, (t => (Ri(), Bi(r, {
                        key: t,
                        name: t,
                        path: n.path.concat(t),
                        data: n.data[t],
                        "expand-signal": a.innerExpandSignal,
                        "collapse-signal": a.innerCollapseSignal,
                        expandOnCreatedAndUpdated: () => !1,
                        getKeys: n.getKeys
                    }, null, 8, ["name", "path", "data", "expand-signal", "collapse-signal", "expandOnCreatedAndUpdated", "getKeys"])))), 128))])) : Hi("v-if", !0)], 64)) : (Ri(), Bi(Oi, {
                        key: 1
                    }, [a.isExpanding ? (Ri(), Bi("span", yr, [(Ri(!0), Bi(Oi, null, _o(a.keys, (t => (Ri(), Bi(r, {
                        key: t,
                        name: t,
                        path: n.path.concat(t),
                        data: n.data[t],
                        "expand-signal": a.innerExpandSignal,
                        "collapse-signal": a.innerCollapseSignal,
                        expandOnCreatedAndUpdated: n.expandOnCreatedAndUpdated,
                        getKeys: n.getKeys
                    }, null, 8, ["name", "path", "data", "expand-signal", "collapse-signal", "expandOnCreatedAndUpdated", "getKeys"])))), 128))])) : Hi("v-if", !0)], 64))])
                }, mr.__file = "src/components/ArrayWrapper.vue";
                var vr = {
                    name: "object-wrapper",
                    props: {
                        path: {
                            required: !0,
                            validator: t => "Array" === Zo(t) && t.every((t => "String" === Zo(t) || "Number" === Zo(t)))
                        },
                        data: {
                            required: !0,
                            validator: t => "Object" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        },
                        collapseSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandOnCreatedAndUpdated: {
                            required: !0,
                            type: Function
                        },
                        getKeys: {
                            required: !0,
                            type: Function
                        }
                    },
                    setup(t) {
                        const {
                            isExpanding: e,
                            innerExpandSignal: n,
                            innerCollapseSignal: a,
                            handleClick: i
                        } = gr(t), o = bo((() => t.getKeys(t.data, t.path))), r = fr.has(t.data);
                        return fr.add(t.data), {
                            keys: o,
                            isExpanding: e,
                            innerExpandSignal: n,
                            innerCollapseSignal: a,
                            handleClick: i,
                            isCircular: r
                        }
                    },
                    components: {}
                };
                const br = {
                        class: "object"
                    },
                    _r = {
                        key: 0,
                        class: "value"
                    },
                    Lr = {
                        key: 1,
                        class: "value"
                    };
                vr.render = function(t, e, n, a, i, o) {
                    const r = Ni("wrapper");
                    return Ri(), Bi("span", br, [Zi("span", {
                        class: "indicator",
                        onClick: e[1] || (e[1] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se(a.isExpanding ? "▼" : "▶"), 1), Zi("span", {
                        class: "key",
                        onClick: e[2] || (e[2] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se("" === n.name ? "" : n.name), 1), Zi("span", {
                        class: "separator",
                        onClick: e[3] || (e[3] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se("" === n.name ? "" : ": "), 1), Zi("span", {
                        class: "preview",
                        onClick: e[4] || (e[4] = (...t) => a.handleClick && a.handleClick(...t))
                    }, se(a.isExpanding ? "" : "{...}"), 1), a.isCircular ? (Ri(), Bi(Oi, {
                        key: 0
                    }, [a.isExpanding ? (Ri(), Bi("span", _r, [(Ri(!0), Bi(Oi, null, _o(a.keys, (t => (Ri(), Bi(r, {
                        key: t,
                        class: "value",
                        name: t,
                        path: n.path.concat(t),
                        data: n.data[t],
                        "expand-signal": a.innerExpandSignal,
                        "collapse-signal": a.innerCollapseSignal,
                        expandOnCreatedAndUpdated: () => !1,
                        getKeys: n.getKeys
                    }, null, 8, ["name", "path", "data", "expand-signal", "collapse-signal", "expandOnCreatedAndUpdated", "getKeys"])))), 128))])) : Hi("v-if", !0)], 64)) : hi((Ri(), Bi("span", Lr, [(Ri(!0), Bi(Oi, null, _o(a.keys, (t => (Ri(), Bi(r, {
                        key: t,
                        class: "value",
                        name: t,
                        path: n.path.concat(t),
                        data: n.data[t],
                        "expand-signal": a.innerExpandSignal,
                        "collapse-signal": a.innerCollapseSignal,
                        expandOnCreatedAndUpdated: n.expandOnCreatedAndUpdated,
                        getKeys: n.getKeys
                    }, null, 8, ["name", "path", "data", "expand-signal", "collapse-signal", "expandOnCreatedAndUpdated", "getKeys"])))), 128))], 512)), [
                        [jo, a.isExpanding]
                    ])])
                }, vr.__file = "src/components/ObjectWrapper.vue";
                const wr = {
                    name: "wrapper",
                    props: {
                        path: {
                            required: !0,
                            validator: t => "Array" === Zo(t) && t.every((t => "String" === Zo(t) || "Number" === Zo(t)))
                        },
                        data: {
                            required: !0,
                            validator: t => "Null" === Zo(t) || "Boolean" === Zo(t) || "Number" === Zo(t) || "String" === Zo(t) || "Array" === Zo(t) || "Object" === Zo(t)
                        },
                        name: {
                            required: !0,
                            type: String
                        },
                        collapseSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandSignal: {
                            default: !1,
                            type: Boolean
                        },
                        expandOnCreatedAndUpdated: {
                            required: !0,
                            type: Function
                        },
                        getKeys: {
                            required: !0,
                            type: Function
                        }
                    },
                    setup: () => ({
                        objectToString: Zo
                    }),
                    components: {
                        NullWrapper: Vo,
                        BooleanWrapper: Jo,
                        NumberWrapper: er,
                        StringWrapper: rr,
                        ArrayWrapper: mr,
                        ObjectWrapper: vr
                    }
                };
                mr.components.Wrapper = wr, vr.components.Wrapper = wr, wr.render = function(t, e, n, a, i, o) {
                    const r = Ni("null-wrapper"),
                        l = Ni("boolean-wrapper"),
                        s = Ni("number-wrapper"),
                        c = Ni("string-wrapper"),
                        u = Ni("array-wrapper"),
                        d = Ni("object-wrapper");
                    return "Null" === t.objectToString(t.data) ? (Ri(), Bi(r, {
                        key: 0,
                        name: t.name,
                        data: t.data
                    }, null, 8, ["name", "data"])) : "Boolean" === t.objectToString(t.data) ? (Ri(), Bi(l, {
                        key: 1,
                        name: t.name,
                        data: t.data
                    }, null, 8, ["name", "data"])) : "Number" === t.objectToString(t.data) ? (Ri(), Bi(s, {
                        key: 2,
                        name: t.name,
                        data: t.data
                    }, null, 8, ["name", "data"])) : "String" === t.objectToString(t.data) ? (Ri(), Bi(c, {
                        key: 3,
                        name: t.name,
                        data: t.data
                    }, null, 8, ["name", "data"])) : "Array" === t.objectToString(t.data) ? (Ri(), Bi(u, {
                        key: 4,
                        name: t.name,
                        path: t.path,
                        data: t.data,
                        "collapse-signal": t.collapseSignal,
                        "expand-signal": t.expandSignal,
                        expandOnCreatedAndUpdated: t.expandOnCreatedAndUpdated,
                        getKeys: t.getKeys
                    }, null, 8, ["name", "path", "data", "collapse-signal", "expand-signal", "expandOnCreatedAndUpdated", "getKeys"])) : "Object" === t.objectToString(t.data) ? (Ri(), Bi(d, {
                        key: 5,
                        name: t.name,
                        path: t.path,
                        data: t.data,
                        "collapse-signal": t.collapseSignal,
                        "expand-signal": t.expandSignal,
                        expandOnCreatedAndUpdated: t.expandOnCreatedAndUpdated,
                        getKeys: t.getKeys
                    }, null, 8, ["name", "path", "data", "collapse-signal", "expand-signal", "expandOnCreatedAndUpdated", "getKeys"])) : Hi("v-if", !0)
                }, wr.__file = "src/components/Wrapper.vue";
                const xr = Object.freeze({
                    expandOnCreatedAndUpdated: t => !1,
                    getKeys: (t, e) => Object.keys(t)
                });
                var kr = gt({
                    components: {
                        CodeAce: ht,
                        CodeMonaco: Kt,
                        FullCodePreview: Yt,
                        Preview: Ht
                    },
                    data: function() {
                        return {
                            mousedown: !1,
                            leftContainerSize: 40,
                            mobileMode: !1,
                            shared: ut,
                            initialCode: "",
                            currentTab: "code-editor",
                            fullCode: "",
                            fullCodeConfig: {
                                mimimal: !1,
                                esm: !0,
                                node: !1
                            }
                        }
                    },
                    computed: {
                        currentTime: function() {
                            this.shared.message;
                            for (var t = new Date, e = [t.getHours(), t.getMinutes(), t.getSeconds()], n = "", a = 0, i = e.length; a < i; ++a) n += (e[a] < 10 ? "0" : "") + e[a], a < i - 1 && (n += ":");
                            return n
                        }
                    },
                    mounted: function() {
                        var t = this;
                        ut.isMobile ? (this.leftContainerSize = 0, dt().then((function(t) {
                            ut.runCode = pt(t)
                        }))) : (dt().then((function(e) {
                            t.initialCode = pt(e)
                        })), window.addEventListener("mousemove", (function(e) {
                            if (t.mousedown) {
                                var n = e.clientX / window.innerWidth;
                                n = Math.min(.9, Math.max(.1, n)), t.leftContainerSize = 100 * n
                            }
                        })), window.addEventListener("mouseup", (function(e) {
                            t.mousedown = !1
                        })))
                    },
                    methods: {
                        onSplitterDragStart: function() {
                            this.mousedown = !0
                        },
                        disposeAndRun: function() {
                            this.$refs.preview.refreshAll()
                        },
                        updateFullCode: function() {
                            var t = this.$refs.preview.getOption();
                            if (t) {
                                var e = (0, Qt.collectDeps)(t);
                                e.push("svg" === ut.renderer ? "SVGRenderer" : "CanvasRenderer"), this.fullCode = (0, Qt.buildExampleCode)(ut.sourceCode, e, {
                                    minimal: this.fullCodeConfig.minimal,
                                    ts: !1,
                                    esm: this.fullCodeConfig.esm,
                                    theme: ut.darkMode ? "dark" : "",
                                    ROOT_PATH: ut.cdnRoot
                                })
                            }
                        },
                        updateOptionOutline: function() {
                            var t = Object.freeze(this.$refs.preview.getOption());
                            t && ((t, e, n = {}) => {
                                void 0 === n.rootName && (n.rootName = ""), void 0 === n.getKeys && (n.getKeys = xr.getKeys), void 0 === n.expandOnCreatedAndUpdated && (n.expandOnCreatedAndUpdated = xr.expandOnCreatedAndUpdated), e.classList.add("object-visualizer"), ((...t) => {
                                    Uo().render(...t)
                                })(null, e), ((...t) => {
                                    const e = Uo().createApp(...t),
                                        {
                                            mount: n
                                        } = e;
                                    return e.mount = t => {
                                        const a = function(t) {
                                            return ke(t) ? document.querySelector(t) : t
                                        }(t);
                                        if (!a) return;
                                        const i = e._component;
                                        xe(i) || i.render || i.template || (i.template = a.innerHTML), a.innerHTML = "";
                                        const o = n(a);
                                        return a.removeAttribute("v-cloak"), a.setAttribute("data-v-app", ""), o
                                    }, e
                                })(wr, {
                                    data: t,
                                    name: n.rootName,
                                    path: [],
                                    expandOnCreatedAndUpdated: n.expandOnCreatedAndUpdated,
                                    getKeys: n.getKeys
                                }).mount(e)
                            })(t, this.$el.querySelector("#option-outline"), {
                                getKeys: function(t, e) {
                                    return Object.keys(t).filter((function(e) {
                                        return !(Array.isArray(t[e]) && !t[e].length)
                                    }))
                                },
                                expandOnCreatedAndUpdated: function(t) {
                                    return 0 === t.length || "series" === t[0] && t.length <= 1
                                }
                            })
                        },
                        updateTabContent: function(t) {
                            "full-code" === t ? this.updateFullCode() : "full-option" === t && this.updateOptionOutline()
                        }
                    },
                    watch: {
                        "shared.typeCheck": function(t) {
                            this.initialCode = ut.sourceCode, this.updateFullCode()
                        },
                        currentTab: function(t) {
                            this.updateTabContent(t)
                        },
                        "shared.runHash": function() {
                            this.updateTabContent(this.currentTab)
                        },
                        fullCodeConfig: {
                            deep: !0,
                            handler: function() {
                                this.updateFullCode()
                            }
                        }
                    }
                }, Q, [], !1, null, null, null);
                kr.options.__file = "src/editor/Editor.vue";
                const Sr = kr.exports;
                var Nr = function() {
                    var t = this,
                        e = t.$createElement,
                        n = t._self._c || e;
                    return n("div", {
                        attrs: {
                            id: "example-explore"
                        }
                    }, [n("div", {
                        attrs: {
                            id: "left-container"
                        }
                    }, [n("div", {
                        attrs: {
                            id: "left-chart-nav"
                        }
                    }, [n("scrollactive", {
                        attrs: {
                            "active-class": "active",
                            offset: 80,
                            duration: 500,
                            "scroll-container-selector": "#example-explore",
                            "bezier-easing-value": ".5,0,.35,1"
                        },
                        on: {
                            itemchanged: t.onActiveNavChanged
                        }
                    }, [n("ul", t._l(t.EXAMPLE_CATEGORIES, (function(e) {
                        return n("li", {
                            key: e
                        }, [n("a", {
                            staticClass: "left-chart-nav-link scrollactive-item",
                            attrs: {
                                id: "left-chart-nav-" + e,
                                href: "#chart-type-" + e
                            }
                        }, [n("span", {
                            staticClass: "chart-icon",
                            domProps: {
                                innerHTML: t._s(t.icons[e])
                            }
                        }), t._v(" "), n("span", {
                            staticClass: "chart-name"
                        }, [t._v(t._s(t.$t("chartTypes." + e)))])])])
                    })), 0)])], 1)]), t._v(" "), n("div", {
                        attrs: {
                            id: "explore-container"
                        }
                    }, [n("div", {
                        staticClass: "example-list-panel"
                    }, t._l(t.exampleList, (function(e) {
                        return n("div", {
                            key: e.category
                        }, [n("h3", {
                            staticClass: "chart-type-head",
                            attrs: {
                                id: "chart-type-" + e.category
                            }
                        }, [t._v("\n                    " + t._s(t.$t("chartTypes." + e.category)) + "\n                    "), n("span", [t._v(t._s(e.category))])]), t._v(" "), n("div", {
                            staticClass: "row",
                            attrs: {
                                id: "chart-row-" + e.category
                            }
                        }, t._l(e.examples, (function(t) {
                            return n("div", {
                                key: t.id,
                                staticClass: "col-xl-2 col-lg-3 col-md-4 col-sm-6"
                            }, [n("ExampleCard", {
                                attrs: {
                                    example: t
                                }
                            })], 1)
                        })), 0)])
                    })), 0)]), t._v(" "), n("div", {
                        attrs: {
                            id: "toolbar"
                        }
                    }, [n("el-switch", {
                        attrs: {
                            "active-color": "#181432",
                            "active-text": t.$t("editor.darkMode"),
                            "inactive-text": ""
                        },
                        model: {
                            value: t.shared.darkMode,
                            callback: function(e) {
                                t.$set(t.shared, "darkMode", e)
                            },
                            expression: "shared.darkMode"
                        }
                    })], 1)])
                };
                Nr._withStripped = !0;
                var Mr = function() {
                    var t = this,
                        e = t.$createElement,
                        a = t._self._c || e;
                    return a("div", {
                        staticClass: "example-list-item"
                    }, [a("a", {
                        staticClass: "example-link",
                        attrs: {
                            target: "_blank",
                            href: t.exampleLink
                        }
                    }, [a("img", {
                        staticClass: "chart-area",
                        attrs: {
                            src: n(555),
                            "data-src": t.screenshotURL
                        }
                    }), t._v(" "), a("h4", {
                        staticClass: "example-title"
                    }, [t._v(t._s(t.title))]), t._v(" "), t.showSubtitle ? a("h5", {
                        staticClass: "example-subtitle"
                    }, [t._v(t._s(t.subtitle))]) : t._e()])])
                };
                Mr._withStripped = !0;
                var Tr = gt({
                    props: ["example"],
                    computed: {
                        title: function() {
                            return ("zh" === ut.locale ? this.example.titleCN : this.example.title) || this.example.title || ""
                        },
                        showSubtitle: function() {
                            return "zh" === ut.locale
                        },
                        subtitle: function() {
                            return this.example.title || ""
                        },
                        exampleTheme: function() {
                            return this.example.theme || (ut.darkMode ? "dark" : "")
                        },
                        exampleLink: function() {
                            var t = this.example,
                                e = ["c=" + t.id],
                                n = this.exampleTheme;
                            return t.isGL && e.push("gl=1"), n && e.push("theme=" + n), "local" in rt && e.push("local"), "useDirtyRect" in rt && e.push("useDirtyRect"), "./editor.html?" + e.join("&")
                        },
                        screenshotURL: function() {
                            var t = this.example,
                                e = this.exampleTheme ? "-" + this.exampleTheme : "",
                                n = st ? "webp" : "png",
                                a = t.isGL ? "data-gl" : "data";
                            return "".concat(ut.cdnRoot, "/").concat(a, "/thumb").concat(e, "/").concat(t.id, ".").concat(n, "?_v_=").concat(ut.version)
                        }
                    }
                }, Mr, [], !1, null, null, null);
                Tr.options.__file = "src/explore/ExampleCard.vue";
                const Or = Tr.exports,
                    Dr = "undefined" != typeof window,
                    Ar = Dr && !("onscroll" in window) || "undefined" != typeof navigator && /(gle|ing|ro)bot|crawl|spider/i.test(navigator.userAgent),
                    Er = Dr && "IntersectionObserver" in window,
                    Pr = Dr && "classList" in document.createElement("p"),
                    Fr = {
                        elements_selector: "img",
                        container: Ar || Dr ? document : null,
                        threshold: 300,
                        thresholds: null,
                        data_src: "src",
                        data_srcset: "srcset",
                        data_sizes: "sizes",
                        data_bg: "bg",
                        data_poster: "poster",
                        class_loading: "loading",
                        class_loaded: "loaded",
                        class_error: "error",
                        load_delay: 0,
                        auto_unobserve: !0,
                        callback_enter: null,
                        callback_exit: null,
                        callback_reveal: null,
                        callback_loaded: null,
                        callback_error: null,
                        callback_finish: null,
                        use_native: !1
                    },
                    Rr = function(t, e) {
                        var n;
                        let a = "LazyLoad::Initialized",
                            i = new t(e);
                        try {
                            n = new CustomEvent(a, {
                                detail: {
                                    instance: i
                                }
                            })
                        } catch (t) {
                            (n = document.createEvent("CustomEvent")).initCustomEvent(a, !1, !1, {
                                instance: i
                            })
                        }
                        window.dispatchEvent(n)
                    },
                    Ir = "data-",
                    Br = "was-processed",
                    jr = "ll-timeout",
                    Gr = "true",
                    zr = (t, e) => t.getAttribute(Ir + e),
                    $r = (t, e, n) => {
                        var a = Ir + e;
                        null !== n ? t.setAttribute(a, n) : t.removeAttribute(a)
                    },
                    Ur = t => zr(t, Br) === Gr,
                    Zr = (t, e) => $r(t, jr, e),
                    Vr = t => zr(t, jr),
                    Wr = (t, e, n, a) => {
                        t && (void 0 === a ? void 0 === n ? t(e) : t(e, n) : t(e, n, a))
                    },
                    Hr = (t, e) => {
                        t.loadingCount += e, 0 === t._elements.length && 0 === t.loadingCount && Wr(t._settings.callback_finish, t)
                    },
                    qr = t => {
                        let e = [];
                        for (let n, a = 0; n = t.children[a]; a += 1) "SOURCE" === n.tagName && e.push(n);
                        return e
                    },
                    Kr = (t, e, n) => {
                        n && t.setAttribute(e, n)
                    },
                    Jr = (t, e) => {
                        Kr(t, "sizes", zr(t, e.data_sizes)), Kr(t, "srcset", zr(t, e.data_srcset)), Kr(t, "src", zr(t, e.data_src))
                    },
                    Xr = {
                        IMG: (t, e) => {
                            const n = t.parentNode;
                            n && "PICTURE" === n.tagName && qr(n).forEach((t => {
                                Jr(t, e)
                            })), Jr(t, e)
                        },
                        IFRAME: (t, e) => {
                            Kr(t, "src", zr(t, e.data_src))
                        },
                        VIDEO: (t, e) => {
                            qr(t).forEach((t => {
                                Kr(t, "src", zr(t, e.data_src))
                            })), Kr(t, "poster", zr(t, e.data_poster)), Kr(t, "src", zr(t, e.data_src)), t.load()
                        }
                    },
                    Yr = (t, e) => {
                        Pr ? t.classList.add(e) : t.className += (t.className ? " " : "") + e
                    },
                    Qr = (t, e) => {
                        Pr ? t.classList.remove(e) : t.className = t.className.replace(new RegExp("(^|\\s+)" + e + "(\\s+|$)"), " ").replace(/^\s+/, "").replace(/\s+$/, "")
                    },
                    tl = "load",
                    el = "loadeddata",
                    nl = "error",
                    al = (t, e, n) => {
                        t.addEventListener(e, n)
                    },
                    il = (t, e, n) => {
                        t.removeEventListener(e, n)
                    },
                    ol = (t, e, n) => {
                        il(t, tl, e), il(t, el, e), il(t, nl, n)
                    },
                    rl = function(t, e, n) {
                        var a = n._settings;
                        const i = e ? a.class_loaded : a.class_error,
                            o = e ? a.callback_loaded : a.callback_error,
                            r = t.target;
                        Qr(r, a.class_loading), Yr(r, i), Wr(o, r, n), Hr(n, -1)
                    },
                    ll = ["IMG", "IFRAME", "VIDEO"],
                    sl = (t, e) => {
                        var n = e._observer;
                        ul(t, e), n && e._settings.auto_unobserve && n.unobserve(t)
                    },
                    cl = t => {
                        var e = Vr(t);
                        e && (clearTimeout(e), Zr(t, null))
                    },
                    ul = (t, e, n) => {
                        var a = e._settings;
                        !n && Ur(t) || (ll.indexOf(t.tagName) > -1 && (((t, e) => {
                            const n = i => {
                                    rl(i, !0, e), ol(t, n, a)
                                },
                                a = i => {
                                    rl(i, !1, e), ol(t, n, a)
                                };
                            ((t, e, n) => {
                                al(t, tl, e), al(t, el, e), al(t, nl, n)
                            })(t, n, a)
                        })(t, e), Yr(t, a.class_loading)), ((t, e) => {
                            const n = e._settings,
                                a = t.tagName,
                                i = Xr[a];
                            if (i) return i(t, n), Hr(e, 1), void(e._elements = (o = e._elements, r = t, o.filter((t => t !== r))));
                            var o, r;
                            ((t, e) => {
                                const n = zr(t, e.data_src),
                                    a = zr(t, e.data_bg);
                                n && (t.style.backgroundImage = `url("${n}")`), a && (t.style.backgroundImage = a)
                            })(t, n)
                        })(t, e), (t => {
                            $r(t, Br, Gr)
                        })(t), Wr(a.callback_reveal, t, e), Wr(a.callback_set, t, e))
                    },
                    dl = t => {
                        return !!Er && (t._observer = new IntersectionObserver((e => {
                            e.forEach((e => (t => t.isIntersecting || t.intersectionRatio > 0)(e) ? ((t, e, n) => {
                                const a = n._settings;
                                Wr(a.callback_enter, t, e, n), a.load_delay ? ((t, e) => {
                                    var n = e._settings.load_delay,
                                        a = Vr(t);
                                    a || (a = setTimeout((function() {
                                        sl(t, e), cl(t)
                                    }), n), Zr(t, a))
                                })(t, n) : sl(t, n)
                            })(e.target, e, t) : ((t, e, n) => {
                                const a = n._settings;
                                Wr(a.callback_exit, t, e, n), a.load_delay && cl(t)
                            })(e.target, e, t)))
                        }), {
                            root: (e = t._settings).container === document ? null : e.container,
                            rootMargin: e.thresholds || e.threshold + "px"
                        }), !0);
                        var e
                    },
                    pl = ["IMG", "IFRAME"],
                    fl = (t, e) => {
                        return (t => t.filter((t => !Ur(t))))((n = t || (t => t.container.querySelectorAll(t.elements_selector))(e), Array.prototype.slice.call(n)));
                        var n
                    },
                    gl = function(t, e) {
                        var n;
                        this._settings = (t => Object.assign({}, Fr, t))(t), this.loadingCount = 0, dl(this), this.update(e), n = this, Dr && window.addEventListener("online", (t => {
                            (t => {
                                var e = t._settings;
                                e.container.querySelectorAll("." + e.class_error).forEach((t => {
                                    Qr(t, e.class_error), (t => {
                                        $r(t, Br, null)
                                    })(t)
                                })), t.update()
                            })(n)
                        }))
                    };
                gl.prototype = {
                    update: function(t) {
                        var e, n = this._settings;
                        this._elements = fl(t, n), !Ar && this._observer ? ((t => t.use_native && "loading" in HTMLImageElement.prototype)(n) && ((e = this)._elements.forEach((t => {
                            -1 !== pl.indexOf(t.tagName) && (t.setAttribute("loading", "lazy"), ul(t, e))
                        })), this._elements = fl(t, n)), this._elements.forEach((t => {
                            this._observer.observe(t)
                        }))) : this.loadAll()
                    },
                    destroy: function() {
                        this._observer && (this._elements.forEach((t => {
                            this._observer.unobserve(t)
                        })), this._observer = null), this._elements = null, this._settings = null
                    },
                    load: function(t, e) {
                        ul(t, this, e)
                    },
                    loadAll: function() {
                        this._elements.forEach((t => {
                            sl(t, this)
                        }))
                    }
                }, Dr && function(t, e) {
                    if (e)
                        if (e.length)
                            for (let n, a = 0; n = e[a]; a += 1) Rr(t, n);
                        else Rr(t, e)
                }(gl, window.lazyLoadOptions);
                const ml = gl;
                var hl = {};
                ["line", "bar", "scatter", "pie", "radar", "funnel", "gauge", "map", "graph", "treemap", "parallel", "sankey", "candlestick", "boxplot", "heatmap", "pictorialBar", "themeRiver", "calendar", "custom", "sunburst", "tree", "dataset", "geo", "lines", "dataZoom", "rich", "drag"].forEach((function(t) {
                    hl[t] = n(472)("./" + t + ".svg")
                }));
                var Cl = n(926);
                ["globe", "bar3D", "scatter3D", "surface", "map3D", "lines3D", "line3D", "scatterGL", "linesGL", "flowGL", "graphGL", "geo3D"].forEach((function(t) {
                    hl[t] = Cl
                }));
                var yl = "ec-shot-loaded",
                    vl = gt({
                        components: {
                            ExampleCard: Or
                        },
                        data: function() {
                            var t = {};

                            function e(e, n) {
                                var a = 0;
                                do {
                                    for (var i = !1, o = 0; o < e.length; o++) {
                                        var r = e[o];
                                        if (!ot.hasOwnProperty(r.id)) {
                                            "string" == typeof r.category && (r.category = [r.category]);
                                            var l = (r.category || [])[a];
                                            if (l) {
                                                i = !0;
                                                var s = t[l];
                                                s || (s = {
                                                    category: l,
                                                    examples: []
                                                }, t[l] = s), r.isGL = n, s.examples.push(r)
                                            }
                                        }
                                    }
                                    if (!i) break
                                } while (++a && a < 4)
                            }
                            return e(Ft, !1), e(Rt, !0), {
                                shared: ut,
                                icons: hl,
                                EXAMPLE_CATEGORIES: it,
                                exampleListByCategory: t
                            }
                        },
                        watch: {
                            "shared.darkMode": function() {
                                for (var t = this.$el.querySelectorAll("img.chart-area"), e = 0; e < t.length; e++) t[e].classList.remove(yl), t[e].setAttribute("data-was-processed", "false");
                                this._lazyload.update()
                            }
                        },
                        computed: {
                            exampleList: function() {
                                for (var t = [], e = 0; e < it.length; e++) {
                                    var n = it[e],
                                        a = this.exampleListByCategory[n];
                                    a && a.examples.length > 0 && t.push({
                                        category: n,
                                        examples: a.examples
                                    })
                                }
                                return t
                            }
                        },
                        mounted: function() {
                            this._lazyload = new ml({
                                elements_selector: "img.chart-area",
                                load_delay: 400,
                                class_loaded: yl
                            })
                        },
                        methods: {
                            onActiveNavChanged: function(t, e, n) {}
                        }
                    }, Nr, [], !1, null, null, null);
                vl.options.__file = "src/explore/Explore.vue";
                const bl = vl.exports;
                var _l = function() {
                    var t = this.$createElement;
                    return (this._self._c || t)("preview")
                };
                _l._withStripped = !0;
                var Ll = gt({
                    components: {
                        Preview: Ht
                    },
                    mounted: function() {
                        dt().then((function(t) {
                            ut.runCode = pt(t)
                        }))
                    }
                }, _l, [], !1, null, null, null);
                Ll.options.__file = "src/editor/View.vue";
                const wl = Ll.exports;
                var xl = n(463),
                    kl = n.n(xl);

                function Sl(t, e) {
                    var n = new X({
                        locale: e.locale,
                        fallbackLocale: "en",
                        messages: Y
                    });
                    if (ut.cdnRoot = e.cdnRoot, ut.version = e.version, ut.locale = e.locale || "en", "string" == typeof t && (t = document.querySelector(t)), !t) throw new Error("Can't find el.");
                    var a = document.createElement("div");
                    t.appendChild(a), new(i())({
                        i18n: n,
                        el: a,
                        render: function(t) {
                            return t({
                                editor: Sr,
                                explore: bl,
                                view: wl
                            }[e.page] || bl)
                        }
                    })
                }
                i().use(kl())
            },
            555: (t, e, n) => {
                t.exports = n.p + "../asset/placeholder.jpg"
            },
            38: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="138px" viewBox="0 0 175 138" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>bar</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="bar" fill="#5067A2" fill-rule="nonzero"> <path d="M121.488231,0 L102.050114,0 C99.36627,-1.1969592e-16 97.1905846,2.17568537 97.1905846,4.85952929 L97.1905846,133.140471 C97.1905846,135.824315 99.36627,138 102.050114,138 L121.488231,138 C124.172075,138 126.34776,135.824315 126.34776,133.140471 L126.34776,4.85952929 C126.34776,2.17568537 124.172075,1.1969592e-16 121.488231,0 Z M170.083523,30.1571753 L150.645406,30.1571753 C147.961562,30.1571753 145.785877,32.3328607 145.785877,35.0167046 L145.785877,132.207289 C145.785877,134.891133 147.961562,137.066819 150.645406,137.066819 L170.083523,137.066819 C172.767367,137.066819 174.943052,134.891133 174.943052,132.207289 L174.943052,35.0167046 C174.943052,32.3328607 172.767367,30.1571753 170.083523,30.1571753 Z M53.4548215,39.8762339 C50.7709777,39.876234 48.5952924,42.0519193 48.5952924,44.7357631 L48.5952924,132.207289 C48.5952924,134.891133 50.7709777,137.066818 53.4548215,137.066819 L72.8929386,137.066819 C75.5767824,137.066818 77.7524677,134.891133 77.7524677,132.207289 L77.7524677,44.7357631 C77.7524676,42.0519193 75.5767824,39.876234 72.8929386,39.8762339 L53.4548215,39.8762339 Z M29.1571753,67.17388 L29.1571753,132.207289 C29.1571753,134.891133 26.9814901,137.066818 24.2976462,137.066819 L4.85952929,137.066819 C2.17568537,137.066819 3.28676086e-16,134.891133 0,132.207289 L0,67.17388 C7.33863613e-08,64.4900361 2.17568542,62.3143508 4.85952929,62.3143508 L24.2976462,62.3143508 C26.98149,62.3143509 29.1571753,64.4900362 29.1571753,67.17388 Z" id="Shape"></path> </g> </g> </svg>'
            },
            851: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="98px" viewBox="0 0 175 98" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>boxplot</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="boxplot" fill="#5067A2" fill-rule="nonzero"> <path d="M165.206074,4.85900217 L165.206074,43.7310195 L145.770065,43.7310195 L145.770065,14.5770065 C145.770065,11.8934537 143.594616,9.71800434 140.911063,9.71800434 L72.8850325,9.71800434 C70.2014797,9.71800434 68.0260304,11.8934537 68.0260304,14.5770065 L68.0260304,82.6030369 C68.0260304,85.2865897 70.2014797,87.462039 72.8850325,87.462039 L140.911063,87.462039 C143.594616,87.462039 145.770065,85.2865897 145.770065,82.6030369 L145.770065,53.4490239 L165.206074,53.4490239 L165.206074,92.3210412 C165.206074,95.004594 167.381523,97.1800434 170.065076,97.1800434 C172.748629,97.1800434 174.924078,95.004594 174.924078,92.3210412 L174.924078,4.85900217 C174.924078,2.17544937 172.748629,1.43751749e-14 170.065076,1.42108547e-14 C167.381523,-1.05827854e-15 165.206074,2.17544937 165.206074,4.85900217 Z M58.308026,14.5770065 L58.308026,82.6030369 C58.308026,83.8917244 57.7960969,85.1276306 56.8848572,86.0388703 C55.9736176,86.9501099 54.7377114,87.462039 53.4490239,87.462039 L34.0130152,87.462039 C31.3294624,87.462039 29.154013,85.2865897 29.154013,82.6030369 L29.154013,53.4490239 L9.71800434,53.4490239 L9.71800434,92.3210412 C9.71800434,95.004594 7.54255497,97.1800434 4.85900217,97.1800434 C2.17544937,97.1800434 3.28640434e-16,95.004594 0,92.3210412 L0,4.85900217 C-3.28640434e-16,2.17544937 2.17544937,4.31255248e-14 4.85900217,4.26325641e-14 C7.54255497,4.21396035e-14 9.71800434,2.17544937 9.71800434,4.85900217 L9.71800434,43.7310195 L29.154013,43.7310195 L29.154013,14.5770065 C29.154013,11.8934537 31.3294624,9.71800434 34.0130152,9.71800434 L53.4490239,9.71800434 C56.1325767,9.71800434 58.308026,11.8934537 58.308026,14.5770065 L58.308026,14.5770065 Z" id="Shape"></path> </g> </g> </svg>'
            },
            496: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="164px" height="138px" viewBox="0 0 164 138" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>calendar </title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="calendar-" transform="translate(-0.500000, -0.315789)" fill="#5067A2" fill-rule="nonzero"> <path d="M147.236842,56.3782895 C147.236842,53.9965615 145.304597,52.0657895 142.921053,52.0657895 L125.657895,52.0657895 C123.27435,52.0657895 121.342105,53.9965615 121.342105,56.3782895 L121.342105,65.0032895 C121.342105,67.3850175 123.27435,69.3157895 125.657895,69.3157895 L142.921053,69.3157895 C145.304597,69.3157895 147.236842,67.3850175 147.236842,65.0032895 L147.236842,56.3782895 Z M147.236842,82.2532895 C147.236842,79.8715615 145.304597,77.9407895 142.921053,77.9407895 L125.657895,77.9407895 C123.27435,77.9407895 121.342105,79.8715615 121.342105,82.2532895 L121.342105,90.8782895 C121.342105,93.2600175 123.27435,95.1907895 125.657895,95.1907895 L142.921053,95.1907895 C145.304597,95.1907895 147.236842,93.2600175 147.236842,90.8782895 L147.236842,82.2532895 Z M112.710526,56.3782895 C112.710526,53.9965615 110.778282,52.0657895 108.394737,52.0657895 L91.1315789,52.0657895 C88.7480342,52.0657895 86.8157895,53.9965615 86.8157895,56.3782895 L86.8157895,65.0032895 C86.8157895,67.3850175 88.7480342,69.3157895 91.1315789,69.3157895 L108.394737,69.3157895 C110.778282,69.3157895 112.710526,67.3850175 112.710526,65.0032895 L112.710526,56.3782895 Z M112.710526,82.2532895 C112.710526,79.8715615 110.778282,77.9407895 108.394737,77.9407895 L91.1315789,77.9407895 C88.7480342,77.9407895 86.8157895,79.8715615 86.8157895,82.2532895 L86.8157895,90.8782895 C86.8157895,93.2600175 88.7480342,95.1907895 91.1315789,95.1907895 L108.394737,95.1907895 C110.778282,95.1907895 112.710526,93.2600175 112.710526,90.8782895 L112.710526,82.2532895 Z M112.710526,108.128289 C112.710526,105.746561 110.778282,103.815789 108.394737,103.815789 L91.1315789,103.815789 C88.7480342,103.815789 86.8157895,105.746561 86.8157895,108.128289 L86.8157895,116.753289 C86.8157895,119.135017 88.7480342,121.065789 91.1315789,121.065789 L108.394737,121.065789 C110.778282,121.065789 112.710526,119.135017 112.710526,116.753289 L112.710526,108.128289 Z M78.1842105,56.3782895 C78.1842105,53.9965615 76.2519658,52.0657895 73.8684211,52.0657895 L56.6052632,52.0657895 C54.2217184,52.0657895 52.2894737,53.9965615 52.2894737,56.3782895 L52.2894737,65.0032895 C52.2894737,67.3850175 54.2217184,69.3157895 56.6052632,69.3157895 L73.8684211,69.3157895 C76.2519658,69.3157895 78.1842105,67.3850175 78.1842105,65.0032895 L78.1842105,56.3782895 Z M78.1842105,82.2532895 C78.1842105,79.8715615 76.2519658,77.9407895 73.8684211,77.9407895 L56.6052632,77.9407895 C54.2217184,77.9407895 52.2894737,79.8715615 52.2894737,82.2532895 L52.2894737,90.8782895 C52.2894737,93.2600175 54.2217184,95.1907895 56.6052632,95.1907895 L73.8684211,95.1907895 C76.2519658,95.1907895 78.1842105,93.2600175 78.1842105,90.8782895 L78.1842105,82.2532895 Z M78.1842105,108.128289 C78.1842105,105.746561 76.2519658,103.815789 73.8684211,103.815789 L56.6052632,103.815789 C54.2217184,103.815789 52.2894737,105.746561 52.2894737,108.128289 L52.2894737,116.753289 C52.2894737,119.135017 54.2217184,121.065789 56.6052632,121.065789 L73.8684211,121.065789 C76.2519658,121.065789 78.1842105,119.135017 78.1842105,116.753289 L78.1842105,108.128289 Z M43.6578947,82.2532895 C43.6578947,79.8715615 41.72565,77.9407895 39.3421053,77.9407895 L22.0789474,77.9407895 C19.6954027,77.9407895 17.7631579,79.8715615 17.7631579,82.2532895 L17.7631579,90.8782895 C17.7631579,93.2600175 19.6954027,95.1907895 22.0789474,95.1907895 L39.3421053,95.1907895 C41.72565,95.1907895 43.6578947,93.2600175 43.6578947,90.8782895 L43.6578947,82.2532895 Z M43.6578947,108.128289 C43.6578947,105.746561 41.72565,103.815789 39.3421053,103.815789 L22.0789474,103.815789 C19.6954027,103.815789 17.7631579,105.746561 17.7631579,108.128289 L17.7631579,116.753289 C17.7631579,119.135017 19.6954027,121.065789 22.0789474,121.065789 L39.3421053,121.065789 C41.72565,121.065789 43.6578947,119.135017 43.6578947,116.753289 L43.6578947,108.128289 Z M164.5,39.1282895 L164.5,134.003289 C164.5,136.385017 162.567755,138.315789 160.184211,138.315789 L4.81578947,138.315789 C2.43224476,138.315789 0.5,136.385017 0.5,134.003289 L0.5,39.1282895 C0.5,36.7465615 2.43224476,34.8157895 4.81578947,34.8157895 L160.184211,34.8157895 C162.567755,34.8157895 164.5,36.7465615 164.5,39.1282895 Z M164.5,4.62828947 L164.5,21.8782895 C164.5,24.2600175 162.567755,26.1907895 160.184211,26.1907895 L4.81578947,26.1907895 C2.43224476,26.1907895 0.5,24.2600175 0.5,21.8782895 L0.5,4.62828947 C0.5,2.24656149 2.43224476,0.315789474 4.81578947,0.315789474 L160.184211,0.315789474 C162.567755,0.315789474 164.5,2.24656149 164.5,4.62828947 Z" id="Shape"></path> </g> </g> </svg>'
            },
            173: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="138px" viewBox="0 0 175 138" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>Candlestick (1)</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="Candlestick-(1)" fill="#5067A2" fill-rule="nonzero"> <path d="M121.508001,34.0222403 L116.647681,33.0222403 L116.647681,4.86032 C116.647681,2.17603934 114.471642,-7.11961547e-08 111.787361,-7.11961549e-08 C109.10308,-7.1196155e-08 106.927041,2.17603934 106.927041,4.86032 L106.927041,33.0222403 L102.066721,33.0222403 C100.777684,33.0222403 99.5414424,33.5343082 98.6299556,34.4457951 C97.7184688,35.3572819 97.2064009,36.5935234 97.2064009,37.8825604 L97.2064009,105.927041 C97.2064009,107.216078 97.7184688,108.452319 98.6299556,109.363806 C99.5414424,110.275293 100.777684,110.787361 102.066721,110.787361 L106.927041,110.787361 L106.927041,125.572455 C106.927041,128.256735 109.10308,130.432775 111.787361,130.432775 C114.471642,130.432775 116.647681,128.256735 116.647681,125.572455 L116.647681,110.787361 L121.508001,110.787361 C122.797038,110.787361 124.03328,110.275293 124.944766,109.363806 C125.856253,108.452319 126.368321,107.216078 126.368321,105.927041 L126.368321,37.8825604 C126.368321,36.5935233 125.856253,35.3572819 124.944766,34.445795 C124.03328,33.5343082 122.797038,33.0222402 121.508001,33.0222403 L121.508001,34.0222403 Z M170.111202,41.6032004 L165.250881,41.6032004 L165.250881,27.0222403 C165.250881,24.3379598 163.074842,22.1619207 160.390561,22.1619207 C157.706281,22.1619207 155.530242,24.3379598 155.530241,27.0222403 L155.530241,41.6032004 L150.669921,41.6032004 C149.380884,41.6032004 148.144643,42.1152684 147.233156,43.0267552 C146.321669,43.938242 145.809601,45.1744835 145.809601,46.4635205 L145.809601,75.6254408 C145.809601,78.3097214 147.985641,80.4857608 150.669921,80.4857608 L155.530241,80.4857608 L155.530241,104.729037 C155.530241,107.413318 157.706281,109.589357 160.390561,109.589357 C163.074842,109.589357 165.250881,107.413318 165.250881,104.729037 L165.250881,80.4857608 L170.111202,80.4857608 C172.795482,80.4857608 174.971522,78.3097214 174.971522,75.6254408 L174.971522,46.4635205 C174.971522,45.1744835 174.459454,43.938242 173.547967,43.0267552 C172.63648,42.1152684 171.400239,41.6032004 170.111202,41.6032004 L170.111202,41.6032004 Z M72.9048007,84.3460808 L68.0444807,84.3460808 L68.0444807,55.1841605 C68.0444805,52.4998799 65.8684412,50.3238407 63.1841606,50.3238407 C60.49988,50.3238407 58.3238406,52.4998799 58.3238405,55.1841605 L58.3238405,84.3460808 L53.4635205,84.3460808 C52.1744835,84.3460807 50.938242,84.8581487 50.0267552,85.7696355 C49.1152683,86.6811224 48.6032003,87.9173639 48.6032004,89.2064009 L48.6032004,118.368321 C48.6032004,119.657358 49.1152684,120.8936 50.0267552,121.805086 C50.938242,122.716573 52.1744835,123.228641 53.4635205,123.228641 L58.3238405,123.228641 L58.3238405,132.949281 C58.3238406,135.633562 60.49988,137.809601 63.1841606,137.809601 C65.8684412,137.809601 68.0444805,135.633562 68.0444807,132.949281 L68.0444807,123.228641 L72.9048007,123.228641 C75.5890813,123.228641 77.7651206,121.052602 77.7651206,118.368321 L77.7651206,89.2064009 C77.7651207,87.9173639 77.2530527,86.6811224 76.3415659,85.7696356 C75.4300791,84.8581488 74.1938377,84.3460808 72.9048007,84.3460808 Z M29.1619203,41.6032004 L29.1619203,90.2064009 C29.1619203,92.8906815 26.9858809,95.0667209 24.3016003,95.0667209 L19.4412801,95.0667209 L19.4412801,119.572455 C19.4412801,122.256735 17.2652407,124.432775 14.5809601,124.432775 C11.8966795,124.432775 9.72064013,122.256735 9.72064013,119.572455 L9.72064013,95.0667209 L4.86032,95.0667209 C2.17603938,95.0667209 3.28729566e-16,92.8906815 0,90.2064009 L0,41.6032004 C-3.28729566e-16,38.9189198 2.17603938,36.7428804 4.86032,36.7428804 L9.72064013,36.7428804 L9.72064013,12.4412801 C9.72064013,9.75699951 11.8966795,7.58096013 14.5809601,7.58096013 C17.2652407,7.58096013 19.4412801,9.75699951 19.4412801,12.4412801 L19.4412801,36.7428804 L24.3016003,36.7428804 C26.9858809,36.7428804 29.1619203,38.9189198 29.1619203,41.6032004 Z" id="Shape"></path> </g> </g> </svg>'
            },
            353: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="159px" height="142px" viewBox="0 0 159 142" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>custom</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="custom" transform="translate(0.000000, -0.825151)" fill="#5067A2" fill-rule="nonzero"> <path d="M74.8235294,81.2918174 L74.8235294,138.091817 C74.8235294,140.705965 72.7298022,142.825151 70.1470588,142.825151 L4.67647054,142.825151 C2.09372718,142.825151 3.16294839e-16,140.705965 0,138.091817 L0,81.2918174 C-3.63362476e-08,80.0364594 0.492698108,78.8325175 1.36970646,77.9448453 C2.24671482,77.0571731 3.43619342,76.558484 4.67647054,76.558484 L70.1470588,76.558484 C71.387336,76.558484 72.5768146,77.0571731 73.4538229,77.9448453 C74.3308313,78.8325175 74.8235294,80.0364594 74.8235294,81.2918174 Z M74.8235294,5.55848402 L74.8235294,62.358484 C74.8235294,63.6138421 74.3308313,64.817784 73.4538229,65.7054562 C72.5768146,66.5931284 71.387336,67.0918174 70.1470588,67.0918174 L4.67647054,67.0918174 C3.43619342,67.0918174 2.24671482,66.5931284 1.36970646,65.7054562 C0.492698108,64.817784 -3.63362423e-08,63.6138421 0,62.358484 L0,5.55848402 C-2.35506485e-15,4.30312598 0.492698161,3.09918414 1.36970651,2.21151195 C2.24671486,1.32383977 3.43619344,0.825150732 4.67647054,0.825150732 L70.1470588,0.825150732 C72.7298022,0.825150732 74.8235294,2.94433623 74.8235294,5.55848402 Z M159,10.2918174 L159,57.6251507 C159,62.8534463 154.812546,67.0918174 149.647059,67.0918174 L93.5294118,67.0918174 C88.363925,67.0918174 84.1764706,62.8534464 84.1764706,57.6251507 L84.1764706,10.2918174 C84.1764706,5.06352179 88.363925,0.825150732 93.5294118,0.825150732 L149.647059,0.825150732 C154.812546,0.825150809 159,5.06352184 159,10.2918174 Z M154.323529,76.558484 C155.563807,76.558484 156.753285,77.0571731 157.630294,77.9448453 C158.507302,78.8325175 159,80.0364594 159,81.2918174 L159,138.091817 C159,140.705965 156.906273,142.825151 154.323529,142.825151 L88.8529412,142.825151 C86.2701978,142.825151 84.1764706,140.705965 84.1764706,138.091817 L84.1764706,81.2918174 C84.1764706,80.0364594 84.6691687,78.8325175 85.5461771,77.9448453 C86.4231854,77.0571731 87.612664,76.558484 88.8529412,76.558484 L154.323529,76.558484 L154.323529,76.558484 Z M93.5294118,133.358484 L149.647059,133.358484 L149.647059,86.0251507 L93.5294118,86.0251507 L93.5294118,133.358484 L93.5294118,133.358484 Z" id="Shape"></path> </g> </g> </svg>'
            },
            6: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="83px" viewBox="0 0 175 83" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>dataZoom</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="dataZoom" fill="#5067A2" fill-rule="nonzero"> <path d="M9.20037003,18.4191593 L9.20957963,13.8143694 L9.20957963,18.4191593 L41.4431082,18.4191593 L41.4431082,64.4670572 L9.20957963,64.4670572 L9.20957963,18.4191593 L9.20037003,18.4191593 Z M133.538904,64.4670572 L133.538904,18.4191593 L165.772433,18.4191593 L165.772433,64.4670572 L133.538904,64.4670572 L133.538904,64.4670572 Z M165.772433,9.20957963 L133.538904,9.20957963 L133.538904,4.60478981 C133.538904,2.06163473 131.47727,2.74466854e-07 128.934115,2.74466854e-07 C126.390959,2.74466855e-07 124.329325,2.06163473 124.329325,4.60478981 L124.329325,9.20957963 L50.6526879,9.20957963 L50.6526879,4.60478981 C50.6526879,2.06163462 48.5910532,-6.1602121e-15 46.047898,-6.31593544e-15 C43.5047429,-6.47165878e-15 41.4431082,2.06163462 41.4431082,4.60478981 L41.4431082,9.20957963 L9.20957963,9.20957963 C4.13510123,9.20957963 0,13.3446809 0,18.4191593 L0,64.4670572 C0,69.5507451 4.13510123,73.6766368 9.20957963,73.6766368 L41.4431082,73.6766368 L41.4431082,78.2814267 C41.4431082,80.8245818 43.5047429,82.8862165 46.047898,82.8862165 C48.5910532,82.8862165 50.6526879,80.8245818 50.6526879,78.2814267 L50.6526879,73.6766368 L124.329325,73.6766368 L124.329325,78.2814267 C124.329325,80.8245818 126.390959,82.8862165 128.934115,82.8862165 C131.47727,82.8862165 133.538904,80.8245818 133.538904,78.2814267 L133.538904,73.6766368 L165.772433,73.6766368 C170.858743,73.6766368 174.982013,69.5533676 174.982013,64.4670572 L174.982013,18.4191593 C174.982013,13.3446809 170.856121,9.20957963 165.772433,9.20957963 Z" id="Shape"></path> </g> </g> </svg>'
            },
            238: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="145px" height="157px" viewBox="0 0 145 157" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>dataset</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="dataset" fill="#5067A2" fill-rule="nonzero"> <path d="M77,133.051622 C77,135.720999 74.7614237,137.884956 72,137.884956 C69.2385763,137.884956 67,135.720999 67,133.051622 L67,92.7182891 C67,90.0489128 69.2385763,87.8849558 72,87.8849558 C74.7614237,87.8849558 77,90.0489128 77,92.7182891 L77,133.051622 Z M87,92.7182888 C87.0000002,90.0489126 89.2385764,87.8849558 92,87.8849558 C94.7614236,87.8849558 96.9999998,90.0489126 97,92.7182888 L97,133.051623 C96.9999998,135.720999 94.7614236,137.884956 92,137.884956 C89.2385764,137.884956 87.0000002,135.720999 87,133.051623 L87,92.7182888 Z M58,133.072456 C58,135.730326 55.7614237,137.884956 53,137.884956 C50.2385763,137.884956 48,135.730326 48,133.072456 L48,73.6974557 C48,71.0395854 50.2385763,68.8849558 53,68.8849558 C55.7614237,68.8849558 58,71.0395854 58,73.6974557 L58,133.072456 Z M39,133.066774 C38.9999999,135.727782 36.7614237,137.884956 34,137.884956 C31.2385763,137.884956 29.0000001,135.727782 29,133.066774 L29,44.7031376 C29,42.0421293 31.2385763,39.8849558 34,39.8849558 C36.7614237,39.8849558 39,42.0421293 39,44.7031376 L39,133.066774 Z M126.3,132.884956 L126.3,63.9913737 C126.403893,62.6917747 125.897136,61.4145164 124.9129,60.4952543 L83.0962,21.7192841 C82.9895,21.6282393 82.8537001,21.5918214 82.7469999,21.5098811 C81.8447692,20.5509026 80.5501183,19.997952 79.1870999,19.9894333 L39,19.9894333 L39,10.8849558 L135.9903,10.8849558 L136,132.884956 L126.3,132.884956 Z M116,146.884956 L10,146.884956 L10,28.8849558 L74.0143636,28.8849558 L74.0143636,58.8841865 C74.0143636,63.8855711 78.3314545,67.9611095 83.6507272,67.9611097 L116,67.9611097 L116,146.884956 Z M144.990333,10.0614263 C144.990333,5.00519104 140.659667,0.884955752 135.323667,0.884955752 L38.6666667,0.884955752 C33.3403334,0.884955752 29.0000001,5.00519104 28.9999999,10.0614263 L28.9999999,19.2378969 L9.66666665,19.2378969 C4.34033332,19.2378969 0,23.3581322 0,28.4143675 L0,147.708485 C0,152.773897 4.34033332,156.884956 9.66666665,156.884956 L116,156.884956 C121.338753,156.884956 125.666667,152.77651 125.666667,147.708485 L125.666667,142.202603 L135.333333,142.202603 C140.672086,142.202603 145,138.094157 145,133.026132 L144.990333,10.0614263 Z" id="Shape"></path> </g> </g> </svg>'
            },
            642: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="159px" viewBox="0 0 175 159" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>drag</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="drag" transform="translate(0.000000, 0.000000)" fill="#5067A2" fill-rule="nonzero"> <path d="M174.799112,77.2243509 C175.141784,80.4933122 173.860538,83.7226514 171.369927,85.8675022 L145.821493,107.665891 C143.704571,109.29545 140.681703,108.978807 138.948417,106.945941 C137.215132,104.913075 137.380357,101.878162 139.32409,100.045479 L158.345038,83.8019696 L92.4385031,83.8019696 L92.4385031,144.173679 L109.273597,130.085944 C111.397298,128.30558 114.562167,128.583909 116.342532,130.70761 C118.122896,132.83131 117.844567,135.99618 115.720867,137.776544 L93.3308935,156.516741 C89.5429956,159.697467 83.9995074,159.633553 80.2859517,156.366338 L59.0490671,137.706356 C57.140269,135.840194 57.0244102,132.80703 58.7852891,130.800695 C60.546168,128.79436 63.5687212,128.515654 65.6667932,130.166159 L82.4116455,144.885586 L82.4116455,83.8019696 L16.5051101,83.8019696 L35.5360859,100.045479 C36.9700273,101.186546 37.6671397,103.019075 37.3541473,104.824694 C37.041155,106.630312 35.7679333,108.121296 34.0335988,108.713163 C32.2992644,109.305031 30.380183,108.903467 29.0286553,107.665891 L3.52032945,85.8975828 C1.25557692,83.9644222 -0.0328273042,81.1246883 0.00397844461,78.147297 C0.0407841935,75.1699056 1.39899211,72.3628885 3.71083975,70.4863026 L29.118897,49.820949 C31.2816477,48.328475 34.2285594,48.7508003 35.8850928,50.7906204 C37.5416262,52.8304405 37.3503183,55.8013071 35.4458442,57.6118174 L15.5625855,73.7751119 L82.4116455,73.7751119 L82.4116455,14.1153089 L65.6667932,28.8347359 C64.3276607,30.0598688 62.43094,30.4655542 60.7078175,29.8954003 C58.9846951,29.3252464 57.7042897,27.8683002 57.360198,26.0862151 C57.0161064,24.30413 57.662069,22.4752334 59.0490671,21.3045658 L80.2759248,2.65461056 C83.9853287,-0.629787191 89.5418994,-0.698070701 93.3308935,2.49418084 L115.720867,21.2243509 C117.320514,22.5881211 117.904229,24.8023516 117.184781,26.777481 C116.465334,28.7526104 114.594242,30.0726692 112.492218,30.0880931 C111.314421,30.0896881 110.174126,29.6740664 109.273597,28.9149508 L92.4385031,14.8272158 L92.4385031,73.7751119 L159.29759,73.7751119 L139.414331,57.6118174 C138.022627,56.4816171 137.339593,54.6948034 137.622523,52.9244518 C137.905453,51.1541001 139.111362,49.669169 140.785997,49.0290176 C142.460631,48.3888662 144.349574,48.6907487 145.741278,49.820949 L171.149336,70.4863026 C173.218763,72.1532673 174.533435,74.5803549 174.799112,77.2243509" id="Path"></path> </g> </g> </svg>'
            },
            797: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="137px" viewBox="0 0 175 137" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>funnel </title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="funnel-" fill="#5067A2" fill-rule="nonzero"> <path d="M116.331467,108.325366 L60.5285591,108.325366 C58.9291301,108.325102 57.4459378,109.160985 56.6178211,110.52934 C55.7897045,111.897695 55.7371711,113.5994 56.4793063,115.016231 L66.3876358,133.863994 C67.1750739,135.367321 68.7306799,136.310539 70.4277481,136.313655 L106.432278,136.313655 C108.129346,136.310539 109.684952,135.367321 110.47239,133.863994 L120.380719,115.016231 C121.122854,113.5994 121.070321,111.897695 120.242204,110.52934 C119.414088,109.160985 117.930896,108.325102 116.331467,108.325366 L116.331467,108.325366 Z M134.968998,71.0137414 L41.9001683,71.0137414 C40.2925309,71.0025161 38.7981975,71.8401445 37.9688482,73.2173896 C37.139499,74.5946346 37.098146,76.3072185 37.8600561,77.722887 L47.8415098,96.5706503 C48.6348072,98.0648166 50.1899261,98.9971843 51.881622,98.9928896 L125.024106,98.9928896 C126.716211,98.9981743 128.270437,98.0607046 129.055078,96.5615098 L139.00911,77.7137465 C139.754942,76.2976526 139.705659,74.5945859 138.879188,73.2239903 C138.052717,71.8533948 136.569493,71.014997 134.968998,71.0137414 L134.968998,71.0137414 Z M153.63395,35.5759245 L23.2352156,35.5759245 C21.6347203,35.5771801 20.151496,36.4155778 19.325025,37.7861734 C18.498554,39.156769 18.4492717,40.8598357 19.1951033,42.2759296 L29.1491354,61.1145524 C29.9359394,62.6172637 31.4930184,63.5578521 33.1892477,63.5550727 L143.70734,63.5550727 C145.402425,63.5537423 146.957543,62.6143387 147.747452,61.1145524 L157.674063,42.266789 C158.415446,40.8514003 158.363824,39.1515952 157.537917,37.7838041 C156.712011,36.4160129 155.231751,35.5788553 153.63395,35.5759245 L153.63395,35.5759245 Z M174.995358,4.69922974 C174.994253,5.44166711 174.81229,6.17265769 174.465208,6.82897225 L164.538597,25.6767355 C163.746223,27.1729824 162.191592,28.108575 160.498485,28.1081153 L14.4968732,28.1081153 C12.8031994,28.1106197 11.2475937,27.1744407 10.4567611,25.6767355 L0.530150516,6.82897225 C-0.215681017,5.41287837 -0.166398756,3.70981169 0.660072225,2.33921611 C1.48654321,0.96862053 2.96976742,0.130222778 4.57026264,0.1289671 L170.425096,0.1289671 C172.949182,0.1289671 174.995358,2.17514338 174.995358,4.69922974 L174.995358,4.69922974 Z" id="Shape"></path> </g> </g> </svg>'
            },
            822: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="100px" viewBox="0 0 175 100" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>gange</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="gange" fill="#5067A2" fill-rule="nonzero"> <path d="M143.698895,82.9089367 C142.437361,67.6655629 135.026556,53.5920156 123.171797,43.9267897 L135.7146,25.9137777 C153.409366,39.7333892 164.305583,60.4959162 165.625777,82.9089367 L143.698895,82.9089367 L143.698895,82.9089367 Z M115.601912,38.6960173 C112.342946,36.8541942 108.909619,35.339582 105.352176,34.174346 L107.654453,22.8655635 C107.981818,21.2534161 107.424245,19.5890834 106.191766,18.4995036 C104.959288,17.4099237 103.239147,17.0606298 101.679304,17.5831975 C100.119461,18.1057651 98.9568933,19.4208039 98.6295287,21.0329513 L96.4193432,31.8996968 C93.4676767,31.3974455 90.4804994,31.132603 87.4865101,31.1077136 C77.1817177,31.1042083 67.0761605,33.9468942 58.284434,39.3222365 L45.5021944,21.4934066 C70.6181595,5.43266895 102.701308,5.12868193 128.117087,20.7106326 L115.601912,38.6960173 L115.601912,38.6960173 Z M31.2649159,82.9089367 L9.34724295,82.9089367 C10.6455448,61.0475255 21.0636813,40.7358347 38.0612364,26.9267794 L50.843476,44.7464001 C39.5322634,54.3814577 32.493597,68.101229 31.2649159,82.9089367 L31.2649159,82.9089367 Z M87.4865101,0.0269797767 C39.2492112,0.0269797767 0,39.276191 0,87.5134899 C5.46353103e-08,90.0565144 2.06152873,92.118043 4.60455321,92.118043 L35.685287,92.118043 C38.2283115,92.118043 40.2898401,90.0565144 40.2898401,87.5134899 C40.315223,61.4580101 61.4310302,40.3422029 87.4865101,40.31682 C89.908505,40.31682 92.2384089,40.6759751 94.5683128,41.0351302 L86.9247546,78.6451205 C82.0531374,79.4923583 78.2774037,83.5443651 78.2774037,88.6646282 C78.2774037,94.3864333 82.9158433,99.0248728 88.6376484,99.0248728 C94.3594535,99.0248728 98.997893,94.3864333 98.997893,88.6646282 C98.997893,85.7361324 97.7546637,83.102328 95.793124,81.2236703 L103.501146,43.3005704 C121.633876,49.8942906 134.68318,67.1337376 134.68318,87.5134899 C134.68318,90.0565144 136.744709,92.118043 139.287733,92.118043 L170.368467,92.118043 C172.911491,92.118043 174.97302,90.0565144 174.97302,87.5134899 C174.97302,39.276191 135.723809,0.0269798756 87.4865101,0.0269797767 L87.4865101,0.0269797767 Z" id="Shape"></path> </g> </g> </svg>'
            },
            317: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="142px" viewBox="0 0 175 142" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>geo</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="geo" fill="#5067A2" fill-rule="nonzero"> <path d="M9.20332621,34.019885 L55.2181167,56.861627 L55.2181167,128.571076 L9.20332621,103.207724 L9.20332621,34.019885 Z M171.819596,50.2815119 L151.996424,43.1124076 C149.63126,42.3318353 147.074346,43.5747977 146.227189,45.9169373 C145.380032,48.2590769 146.550177,50.8501245 148.867418,51.7631882 L165.653614,57.8371405 L165.653614,131.138702 L119.638823,119.625801 L119.638823,87.4706656 C119.638823,84.9293389 117.578671,82.8691865 115.037344,82.8691865 C112.496018,82.8691865 110.435865,84.9293389 110.435865,87.4706656 L110.435865,119.644207 L64.4210748,131.41479 L64.4210748,58.2236647 L81.060023,51.7263763 C83.2914479,50.7138845 84.3410975,48.1328302 83.4496557,45.8503474 C82.5582139,43.5678645 80.037126,42.3813901 77.7101463,43.1492194 L61.2460542,49.5820871 L6.64490386,22.4793755 C5.21943693,21.7614379 3.52312776,21.8367966 2.16693957,22.6783095 C0.810751382,23.5198224 -0.00997312356,25.0062796 0.000368118324,26.6023007 L0.000368118324,105.913394 C0.000368118324,107.597535 0.91146097,109.134429 2.38393426,109.944289 L58.8072703,141.059491 C59.8287987,141.620871 61.0435892,141.795727 62.16635,141.49203 L115.046547,127.963681 L169.141535,141.501233 C170.516378,141.840666 171.970958,141.532118 173.089604,140.663763 C174.206625,139.788526 174.858383,138.44767 174.856572,137.028595 L174.856572,54.6069022 C174.855748,52.6695411 173.64154,50.9402146 171.819596,50.2815119 Z M115.037344,16.5986853 C121.13907,16.6012266 126.083435,21.5497125 126.080894,27.651438 C126.078353,33.7531634 121.129867,38.697529 115.028141,38.6949877 C108.926416,38.6924463 103.98205,33.7439604 103.984592,27.642235 C103.987133,21.5405096 108.935619,16.596144 115.037344,16.5986853 L115.037344,16.5986853 Z M99.5763747,54.7173377 C102.208421,58.3341002 105.00612,61.9140509 107.813022,65.199507 C108.806942,66.3866886 109.718035,67.4358258 110.491083,68.3285127 C111.04326,68.8806902 111.402176,69.2856204 111.558626,69.4512736 C112.438859,70.4713636 113.723372,71.0524431 115.07068,71.0400337 C116.417988,71.0276243 117.69158,70.4229833 118.552874,69.3868529 C118.72773,69.2211997 119.06824,68.8254725 119.555997,68.264092 C120.506081,67.2663872 121.406012,66.2220982 122.252463,65.1350863 C125.059366,61.8496302 127.866268,58.2788825 130.498314,54.6437141 C133.121157,51.0177486 135.412694,47.4470008 137.262488,44.0879211 C140.741206,37.8851274 142.646219,32.5382087 142.646219,28.1207888 C142.701436,12.5861956 130.314255,0.0425636812 115.028141,0.0425636812 C99.7604339,0.0425636812 87.42847,12.5861956 87.42847,28.1207888 C87.42847,32.5382087 89.3334823,37.9403451 92.8029975,44.1523418 C94.7172128,47.5114215 97.0087494,51.0729663 99.5763747,54.7173377 Z" id="Shape"></path> </g> </g> </svg>'
            },
            926: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="120px" viewBox="0 0 175 120" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>gl</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="gl" fill="#5067A2"> <path d="M170.394737,101.315789 L119.736842,101.315789 L119.736842,4.60526316 C119.736842,2.06315789 117.673684,0 115.131579,0 L105.921053,0 C103.378947,0 101.315789,2.06315789 101.315789,4.60526316 L101.315789,115.131579 C101.315789,117.673684 103.378947,119.736842 105.921053,119.736842 L170.394737,119.736842 C172.936842,119.736842 175,117.673684 175,115.131579 L175,105.921053 C175,103.378947 172.936842,101.315789 170.394737,101.315789" id="Fill-1"></path> <path d="M78.2894737,0 L4.60526316,0 C2.06315789,0 0,2.06315789 0,4.60526316 L0,115.131579 C0,117.673684 2.06315789,119.736842 4.60526316,119.736842 L78.2894737,119.736842 C80.8315789,119.736842 82.8947368,117.673684 82.8947368,115.131579 L82.8947368,59.8684211 C82.8947368,57.3263158 80.8315789,55.2631579 78.2894737,55.2631579 L41.4473684,55.2631579 C38.9052632,55.2631579 36.8421053,57.3263158 36.8421053,59.8684211 L36.8421053,69.0789474 C36.8421053,71.6210526 38.9052632,73.6842105 41.4473684,73.6842105 L64.4736842,73.6842105 L64.4736842,101.315789 L18.4210526,101.315789 L18.4210526,18.4210526 L64.4736842,18.4210526 L64.4736842,32.2368421 C64.4736842,34.7789474 66.5368421,36.8421053 69.0789474,36.8421053 L78.2894737,36.8421053 C80.8315789,36.8421053 82.8947368,34.7789474 82.8947368,32.2368421 L82.8947368,4.60526316 C82.8947368,2.06315789 80.8315789,0 78.2894737,0" id="Fill-3"></path> </g> </g> </svg>'
            },
            769: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="171px" height="146px" viewBox="0 0 171 146" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>graph</title> <desc>Created with Sketch.</desc> <defs> <polygon id="path-1" points="0.06 0 171 0 171 146 0.06 146"></polygon> </defs> <g id="graph" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <mask id="mask-2" fill="white"> <use xlink:href="#path-1"></use> </mask> <g id="Clip-2"></g> <path d="M171,100 C171,106.075 166.075,111 160,111 C154.016,111 149.158,106.219 149.014,100.27 L114.105,83.503 C111.564,86.693 108.179,89.18 104.282,90.616 L108.698,124.651 C112.951,126.172 116,130.225 116,135 C116,141.075 111.075,146 105,146 C98.925,146 94,141.075 94,135 C94,131.233 95.896,127.912 98.781,125.93 L94.364,91.896 C82.94,90.82 74,81.206 74,69.5 C74,69.479 74.001,69.46 74.001,69.439 L53.719,64.759 C50.642,70.269 44.76,74 38,74 C36.07,74 34.215,73.689 32.472,73.127 L20.624,90.679 C21.499,92.256 22,94.068 22,96 C22,102.075 17.075,107 11,107 C4.925,107 0,102.075 0,96 C0,89.925 4.925,85 11,85 C11.452,85 11.895,85.035 12.332,85.089 L24.184,67.531 C21.574,64.407 20,60.389 20,56 C20,48.496 24.594,42.07 31.121,39.368 L29.111,21.279 C24.958,19.707 22,15.704 22,11 C22,4.925 26.925,0 33,0 C39.075,0 44,4.925 44,11 C44,14.838 42.031,18.214 39.051,20.182 L41.061,38.279 C49.223,39.681 55.49,46.564 55.95,55.011 L76.245,59.694 C79.889,52.181 87.589,47 96.5,47 C100.902,47 105.006,48.269 108.475,50.455 L131.538,27.391 C131.192,26.322 131,25.184 131,24 C131,17.925 135.925,13 142,13 C148.075,13 153,17.925 153,24 C153,30.075 148.075,35 142,35 C140.816,35 139.678,34.808 138.609,34.461 L115.546,57.525 C117.73,60.994 119,65.098 119,69.5 C119,71.216 118.802,72.884 118.438,74.49 L153.345,91.257 C155.193,89.847 157.495,89 160,89 C166.075,89 171,93.925 171,100" id="Fill-1" fill="#4F608A" mask="url(#mask-2)"></path> </g> </svg>'
            },
            781: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="165px" height="137px" viewBox="0 0 165 137" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>heatmap</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="heatmap" transform="translate(-0.500000, -0.474277)" fill-rule="nonzero"> <path d="M44.5999999,89.4742765 L5.40000005,89.4742765 C2.6938048,89.4742765 0.500000074,87.2916647 0.5,84.5992766 L0.5,55.3492765 C0.500000074,52.6568883 2.6938048,50.4742765 5.40000005,50.4742765 L44.5999999,50.4742765 C47.3061952,50.4742765 49.4999999,52.6568883 49.5,55.3492765 L49.5,84.5992766 C49.4999999,87.2916647 47.3061952,89.4742765 44.5999999,89.4742765" id="Path" fill="#687DB0"></path> <path d="M102.6,89.4742765 L63.3999999,89.4742765 C60.6938047,89.4742765 58.5000001,87.2916647 58.5,84.5992766 L58.5,55.3492765 C58.5000001,52.6568884 60.6938047,50.4742766 63.3999999,50.4742765 L102.6,50.4742765 C105.306195,50.4742765 107.5,52.6568883 107.5,55.3492765 L107.5,84.5992766 C107.5,87.2916647 105.306195,89.4742765 102.6,89.4742765" id="Path" fill="#8497C0"></path> <path d="M160.6,89.4742765 L121.4,89.4742765 C118.693805,89.4742765 116.5,87.2916647 116.5,84.5992766 L116.5,55.3492765 C116.5,52.6568884 118.693805,50.4742766 121.4,50.4742765 L160.6,50.4742765 C163.306195,50.4742766 165.5,52.6568884 165.5,55.3492765 L165.5,84.5992766 C165.5,87.2916647 163.306195,89.4742765 160.6,89.4742765" id="Path" fill="#A7B4D1"></path> <path d="M44.5999999,137.474277 L5.40000005,137.474277 C2.69380475,137.474277 0.5,135.291665 0.5,132.599276 L0.5,103.349277 C0.5,100.656888 2.69380475,98.4742765 5.40000005,98.4742765 L44.5999999,98.4742765 C45.8995608,98.4742765 47.1458949,98.9878911 48.0648232,99.902131 C48.9837515,100.816371 49.5,102.056346 49.5,103.349277 L49.5,132.599276 C49.5,135.291665 47.3061953,137.474277 44.5999999,137.474277" id="Path" fill="#8497C0"></path> <path d="M102.6,137.474277 L63.3999999,137.474277 C60.6938047,137.474276 58.5,135.291665 58.5,132.599276 L58.5,103.349277 C58.5,100.656888 60.6938047,98.4742766 63.3999999,98.4742765 L102.6,98.4742765 C105.306195,98.4742765 107.5,100.656888 107.5,103.349277 L107.5,132.599276 C107.5,135.291665 105.306195,137.474277 102.6,137.474277" id="Path" fill="#A7B4D1"></path> <path d="M160.6,137.474277 L121.4,137.474277 C118.693805,137.474276 116.5,135.291665 116.5,132.599276 L116.5,103.349277 C116.5,100.656888 118.693805,98.4742766 121.4,98.4742765 L160.6,98.4742765 C163.306195,98.4742766 165.5,100.656888 165.5,103.349277 L165.5,132.599276 C165.5,135.291665 163.306195,137.474276 160.6,137.474277" id="Path" fill="#CDD5E3"></path> <path d="M5.40000005,0.474276527 L44.5999999,0.474276527 C47.3061953,0.474276527 49.5,2.65688839 49.5,5.34927658 L49.5,34.5992765 C49.5,37.2916647 47.3061953,39.4742765 44.5999999,39.4742765 L5.40000005,39.4742765 C2.69380475,39.4742765 0.5,37.2916647 0.5,34.5992765 L0.5,5.34927658 C0.5,2.65688839 2.69380475,0.474276527 5.40000005,0.474276527" id="Path" fill="#5067A2"></path> <path d="M102.6,39.4742765 L63.3999999,39.4742765 C60.6938047,39.4742765 58.5,37.2916646 58.5,34.5992765 L58.5,5.34927658 C58.5,2.65688845 60.6938047,0.474276601 63.3999999,0.474276527 L102.6,0.474276527 C105.306195,0.474276527 107.5,2.65688839 107.5,5.34927658 L107.5,34.5992765 C107.5,37.2916647 105.306195,39.4742765 102.6,39.4742765" id="Path" fill="#687DB0"></path> <path d="M160.6,39.4742765 L121.4,39.4742765 C118.693805,39.4742765 116.5,37.2916646 116.5,34.5992765 L116.5,5.34927658 C116.5,2.65688845 118.693805,0.474276601 121.4,0.474276527 L160.6,0.474276527 C163.306195,0.474276601 165.5,2.65688845 165.5,5.34927658 L165.5,34.5992765 C165.5,37.2916646 163.306195,39.4742765 160.6,39.4742765" id="Path" fill="#8497C0"></path> </g> </g> </svg>'
            },
            69: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="138px" viewBox="0 0 175 138" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>line</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="line" fill="#5067A2" fill-rule="nonzero"> <path d="M25.2739189,111.655778 C26.7855309,111.657407 28.2007506,110.913642 29.0567232,109.667735 L69.3606163,51.4346379 L108.642876,94.5825372 C109.596174,95.6275458 110.977093,96.1760957 112.387618,96.0700817 C113.798143,95.9640677 115.081554,95.2152689 115.86794,94.0395069 L167.15209,17.582681 C168.126301,16.2187314 168.281888,14.4329668 167.558237,12.9210891 C166.834585,11.4092114 165.346152,10.4103472 163.672804,10.3136363 C161.999456,10.2169253 160.40586,11.0376639 159.512851,12.4561069 L111.496086,84.0348638 L72.2782537,40.9605957 C71.3313898,39.926718 69.968454,39.3754562 68.5690807,39.4603596 C67.1694417,39.5586882 65.8899144,40.2864616 65.0900052,41.4391988 L21.4911146,104.439918 C20.5200574,105.846972 20.4090348,107.676368 21.202809,109.190526 C21.9965833,110.704685 23.5643138,111.65402 25.2739189,111.655778" id="Path"></path> <path d="M170.272214,127.854648 L9.20390349,127.854648 L9.20390349,5.00465881 C9.20390349,2.46307104 7.14353951,0.402707066 4.60195174,0.402707066 C2.06036398,0.402707066 3.11254732e-16,2.46307104 0,5.00465881 L0,132.4566 C3.11254728e-16,134.998188 2.06036395,137.058552 4.60195168,137.058552 L170.272214,137.058552 C172.813801,137.058552 174.874165,134.998188 174.874165,132.4566 C174.874165,129.915012 172.813801,129.257355 170.272214,129.257355" id="Path"></path> </g> </g> </svg>'
            },
            276: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="163px" height="137px" viewBox="0 0 163 137" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>lines</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="lines" fill="#5067A2" fill-rule="nonzero"> <path d="M118.024118,118.8 C120.522388,118.8 122.566427,120.8475 122.566427,123.35 C122.566427,125.8525 120.522388,127.9 118.024118,127.9 C115.517543,127.894993 113.486807,125.860819 113.481809,123.35 C113.481809,120.8475 115.525848,118.8 118.024118,118.8 Z M142.434485,46 L64.9673569,46 C62.4587091,46 60.4250482,48.0371044 60.4250482,50.55 C60.4250482,53.0628956 62.4587091,55.1 64.9673569,55.1 L142.034762,55.0909 C149.519307,56.1087027 154.7743,62.9956941 153.790257,70.4972 C152.963556,76.6852 148.066948,81.5264 142.434485,82.2544 L20.6642736,82.2726 C14.6272756,82.8301001 9.06337697,85.7822231 5.21133949,90.4717 C0.0521031846,96.7506172 -1.39663076,105.298969 1.40493936,112.931618 C4.20650947,120.564268 10.8382585,126.136619 18.8291809,127.5724 L105.232977,127.8909 C107.143393,133.33221 112.265894,136.980155 118.024118,137 C125.537097,137 131.651044,130.8757 131.651044,123.35 C131.651044,115.8243 125.537097,109.7 118.024118,109.7 C112.110032,109.7 107.113492,113.5129 105.232977,118.8 L21.8634431,118.8728 L20.4099043,118.618 C14.3307333,117.521702 9.72082575,112.509885 9.12680957,106.3512 C8.77445503,102.712306 9.88625731,99.0830035 12.2155795,96.2684 C14.5594108,93.4383 17.8571269,91.6911 21.0912506,91.3544 L142.988646,91.3271 C153.280464,90.058092 161.421536,81.9929721 162.802197,71.6984 C164.446513,59.2314 155.679857,47.729 142.434485,46 Z" id="Shape"></path> <path d="M31.9999101,18.2496933 C36.9704563,18.2496933 40.9998801,22.3350263 40.9998801,27.37454 C40.9998801,32.4140537 36.9704563,36.4993867 31.9999101,36.4993867 C27.0293639,36.4993867 22.9999401,32.4140537 22.9999401,27.37454 C22.9999401,22.3350263 27.0293639,18.2496933 31.9999101,18.2496933 Z M16.8799604,56.2181803 C19.4539519,59.9411177 22.1809428,63.6184309 24.9259336,66.9946242 C25.9069304,68.2173537 26.7979274,69.3032104 27.5539249,70.2156951 C28.1029231,70.7814356 28.435922,71.1920537 28.5979214,71.3654258 C30.4699152,73.555389 33.5839048,73.555389 35.4378986,71.3015518 C35.617898,71.1373046 35.932897,70.7266865 36.4188954,70.1518212 C37.2648925,69.2393365 38.1288897,68.1443549 39.0558866,66.9307503 C41.8008774,63.554557 44.5368683,59.8863687 47.1108598,56.1451815 C49.6758512,52.4131192 51.9258437,48.7449309 53.7348377,45.2957388 C57.1368264,38.917471 58.9998202,33.4243133 58.9998202,28.8710148 C59.05382,12.911658 46.9398603,0 31.9909101,0 C17.0509599,0 5,12.911658 5,28.8710148 C5,33.4243133 6.8629938,38.9722201 10.2559825,45.3596128 C12.1279763,48.8088048 14.3689688,52.4678683 16.8799604,56.2181803 Z" id="Shape"></path> </g> </g> </svg>'
            },
            831: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="142px" viewBox="0 0 175 142" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>geo</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="geo" fill="#5067A2" fill-rule="nonzero"> <path d="M9.20332621,34.019885 L55.2181167,56.861627 L55.2181167,128.571076 L9.20332621,103.207724 L9.20332621,34.019885 Z M171.819596,50.2815119 L151.996424,43.1124076 C149.63126,42.3318353 147.074346,43.5747977 146.227189,45.9169373 C145.380032,48.2590769 146.550177,50.8501245 148.867418,51.7631882 L165.653614,57.8371405 L165.653614,131.138702 L119.638823,119.625801 L119.638823,87.4706656 C119.638823,84.9293389 117.578671,82.8691865 115.037344,82.8691865 C112.496018,82.8691865 110.435865,84.9293389 110.435865,87.4706656 L110.435865,119.644207 L64.4210748,131.41479 L64.4210748,58.2236647 L81.060023,51.7263763 C83.2914479,50.7138845 84.3410975,48.1328302 83.4496557,45.8503474 C82.5582139,43.5678645 80.037126,42.3813901 77.7101463,43.1492194 L61.2460542,49.5820871 L6.64490386,22.4793755 C5.21943693,21.7614379 3.52312776,21.8367966 2.16693957,22.6783095 C0.810751382,23.5198224 -0.00997312356,25.0062796 0.000368118324,26.6023007 L0.000368118324,105.913394 C0.000368118324,107.597535 0.91146097,109.134429 2.38393426,109.944289 L58.8072703,141.059491 C59.8287987,141.620871 61.0435892,141.795727 62.16635,141.49203 L115.046547,127.963681 L169.141535,141.501233 C170.516378,141.840666 171.970958,141.532118 173.089604,140.663763 C174.206625,139.788526 174.858383,138.44767 174.856572,137.028595 L174.856572,54.6069022 C174.855748,52.6695411 173.64154,50.9402146 171.819596,50.2815119 Z M115.037344,16.5986853 C121.13907,16.6012266 126.083435,21.5497125 126.080894,27.651438 C126.078353,33.7531634 121.129867,38.697529 115.028141,38.6949877 C108.926416,38.6924463 103.98205,33.7439604 103.984592,27.642235 C103.987133,21.5405096 108.935619,16.596144 115.037344,16.5986853 L115.037344,16.5986853 Z M99.5763747,54.7173377 C102.208421,58.3341002 105.00612,61.9140509 107.813022,65.199507 C108.806942,66.3866886 109.718035,67.4358258 110.491083,68.3285127 C111.04326,68.8806902 111.402176,69.2856204 111.558626,69.4512736 C112.438859,70.4713636 113.723372,71.0524431 115.07068,71.0400337 C116.417988,71.0276243 117.69158,70.4229833 118.552874,69.3868529 C118.72773,69.2211997 119.06824,68.8254725 119.555997,68.264092 C120.506081,67.2663872 121.406012,66.2220982 122.252463,65.1350863 C125.059366,61.8496302 127.866268,58.2788825 130.498314,54.6437141 C133.121157,51.0177486 135.412694,47.4470008 137.262488,44.0879211 C140.741206,37.8851274 142.646219,32.5382087 142.646219,28.1207888 C142.701436,12.5861956 130.314255,0.0425636812 115.028141,0.0425636812 C99.7604339,0.0425636812 87.42847,12.5861956 87.42847,28.1207888 C87.42847,32.5382087 89.3334823,37.9403451 92.8029975,44.1523418 C94.7172128,47.5114215 97.0087494,51.0729663 99.5763747,54.7173377 Z" id="Shape"></path> </g> </g> </svg>'
            },
            582: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="148px" viewBox="0 0 175 148" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>parallel</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="parallel" fill="#5067A2" fill-rule="nonzero"> <path d="M119.966527,90.2642567 L119.966527,71.4595035 L164.953975,91.1040224 L164.953975,117.16675 L119.966527,90.2642567 Z M99.5422259,68.8202399 L109.969317,68.8202399 L109.969317,80.2370544 L99.5422259,68.8202399 Z M64.9818688,58.8230293 L64.9818688,45.8666444 L76.8485579,58.8230293 L64.9818688,58.8230293 L64.9818688,58.8230293 Z M9.99721065,95.6627504 L9.99721065,37.9888424 L54.9846583,37.9888424 L54.9846583,61.3523236 L9.99721065,95.6627504 Z M170,2.04051565e-11 C167.238576,2.04245402e-11 165,2.23857623 165,4.99999995 L165,80.2194419 L120,60.5694419 L120,5.00139509 C120,2.23997133 117.761424,0.00139506489 115,0.00139506489 C112.238576,0.00139506489 110,2.23997133 110,5.00139509 L110,58.8394419 L90.42,58.8394419 L64.9999999,31.059442 L64.9999999,5.00139509 C64.9999999,2.23997137 62.7614237,0.00139513813 60,0.00139513813 C57.2385763,0.00139513813 55,2.23997137 55,5.00139509 L55,27.999442 L10,27.999442 L10,4.99999995 C10,2.23857619 7.76142378,-7.3221713e-08 5.00000002,-7.32217131e-08 C2.23857626,-7.32217133e-08 3.38176877e-16,2.23857619 0,4.99999995 L0,142.999442 C3.38176877e-16,145.760866 2.23857626,147.999442 5.00000002,147.999442 C7.76142378,147.999442 10,145.760866 10,142.999442 L10,108.249442 L55,73.9494419 L55,142.999442 C55,145.760866 57.2385763,147.999442 60,147.999442 C62.7614237,147.999442 64.9999999,145.760866 64.9999999,142.999442 L64.9999999,68.839442 L86.01,68.839442 L110,95.0694419 L110,142.999442 C110,145.760866 112.238576,147.999442 115,147.999442 C117.761424,147.999442 120,145.760866 120,142.999442 L120,101.939442 L165,128.839442 L165,142.999442 C165,145.760866 167.238576,147.999442 170,147.999442 C172.761424,147.999442 175,145.760866 175,142.999442 L175,4.99999995 C175,3.67391749 174.473216,2.40214791 173.535534,1.46446603 C172.597852,0.526784141 171.326082,-3.88296707e-08 170,2.04051565e-11 Z" id="Shape"></path> </g> </g> </svg>'
            },
            689: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="141px" viewBox="0 0 175 141" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>pictorialBar</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="pictorialBar" fill="#5067A2" fill-rule="nonzero"> <path d="M165.182447,113.123194 C161.75364,113.169204 158.605604,115.027549 156.908673,118.007359 L62.4782339,118.007359 C59.7807842,118.007359 57.5940692,120.194074 57.5940692,122.891524 C57.5940692,125.588973 59.7807842,127.775688 62.4782339,127.775688 L156.908673,127.775688 C158.605604,130.755498 161.75364,132.613843 165.182447,132.659853 C170.576885,132.6592 174.949595,128.285961 174.949595,122.891524 C174.949595,117.497086 170.576885,113.123847 165.182447,113.123194 L165.182447,113.123194 Z M165.182447,78.9340419 C161.75364,78.980052 158.605603,80.8383969 156.908673,83.8182066 L77.2674843,83.8182066 C74.5700347,83.8182066 72.3833197,86.0049216 72.3833197,88.7023712 C72.3833197,91.3998208 74.5700347,93.5865358 77.2674843,93.5865358 L156.908673,93.5865358 C158.605603,96.5663455 161.75364,98.4246904 165.182447,98.4707005 C170.577347,98.4707005 174.950777,94.0972705 174.950777,88.7023712 C174.950777,83.3074719 170.577347,78.9340419 165.182447,78.9340419 L165.182447,78.9340419 Z M165.182447,44.7448896 C170.407595,44.9786234 174.523583,49.2828462 174.523583,54.5132188 C174.523583,59.7435915 170.407595,64.0478142 165.182447,64.2815481 C161.75364,64.235538 158.605604,62.3771932 156.908673,59.3973835 L77.2674843,59.3973835 C74.5700346,59.3973835 72.3833196,57.2106685 72.3833196,54.5132188 C72.3833196,51.8157692 74.5700346,49.6290542 77.2674843,49.6290542 L156.908673,49.6290542 C158.605604,46.6492445 161.75364,44.7908997 165.182447,44.7448896 Z M62.6149904,20.3240665 C62.6149904,19.0287054 63.1295706,17.7863991 64.0455291,16.8704405 C64.9614877,15.954482 66.2037941,15.4399018 67.4991551,15.4399019 L156.908673,15.4399019 C158.605604,12.4600922 161.75364,10.6017474 165.182447,10.5557373 C170.407595,10.7894711 174.523583,15.0936938 174.523583,20.3240665 C174.523583,25.5544392 170.407595,29.8586619 165.182447,30.0923957 C161.75364,30.0463857 158.605604,28.1880408 156.908673,25.2082312 L67.4991551,25.2082312 C66.2037941,25.2082312 64.9614877,24.6936511 64.0455291,23.7776925 C63.1295705,22.8617339 62.6149904,21.6194275 62.6149904,20.3240665 Z M23.5416735,29.3240665 C31.6340224,29.3240665 38.1941674,22.7639215 38.1941674,14.6715726 C38.1941674,6.57922368 31.6340224,0.0190787204 23.5416735,0.0190787204 C15.4493246,0.0190787204 8.88917966,6.57922368 8.88917966,14.6715726 C8.88917966,22.7639215 15.4493246,29.3240665 23.5416735,29.3240665 Z M34.0523957,34.2082312 L13.0211829,34.2082312 C10.6337153,34.2093302 8.59694107,35.9362503 8.20539657,38.2913927 L0.0683783524,87.133039 C-0.166957522,88.5485838 0.232108973,89.996 1.15960938,91.0909397 C2.08710979,92.1858793 3.44919091,92.8175461 4.88416457,92.8182066 L8.88917966,92.8182066 L9.0426883,136.115835 C9.0426883,138.813285 11.2294033,141 13.9268529,141 L33.4635113,141 C34.7588723,141 36.0011787,140.48542 36.9171373,139.569461 C37.8330959,138.653503 38.347676,137.411196 38.347676,136.115835 L38.1941674,92.8182066 L42.189414,92.8182066 C43.6255482,92.8209447 44.9898051,92.1902296 45.9180572,91.0944026 C46.8463094,89.9985755 47.2440794,88.5491696 47.0052003,87.133039 L38.8681821,38.2913927 C38.4766376,35.9362503 36.4398634,34.2093302 34.0523957,34.2082312 Z" id="Shape"></path> </g> </g> </svg>'
            },
            931: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="144px" height="152px" viewBox="0 0 144 152" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>pie (2)</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="pie-(2)" transform="translate(0.000000, -0.612903)" fill="#5067A2" fill-rule="nonzero"> <path d="M72.1778824,12.9606446 L72.1778824,73.6462877 C72.1779317,75.129331 72.9386663,76.5065116 74.1881954,77.2856176 C75.4377246,78.0647237 76.9977039,78.1345538 78.3105882,77.4701504 L128.354824,52.1402676 L132.666353,49.9503148 C134.757503,48.8920984 137.302156,49.7461658 138.350118,51.857969 L138.468706,52.1060495 L138.782118,52.8930638 C140.942118,58.6416896 144,65.8873535 144,83.8518095 C144,116.316148 115.123765,152.612903 73.2282353,152.612903 C31.3242354,152.612903 0,119.994584 0,80.8406245 C0,44.5353146 27.216,13.7391044 62.2588236,9.22232682 C64.122353,8.98280072 65.9265883,8.80315611 67.6715294,8.69194766 C68.8395088,8.61630135 69.986418,9.03213213 70.8400527,9.84074781 C71.6936873,10.6493635 72.1779411,11.7786714 72.1778824,12.9606446 Z M90.0254118,0.659269609 C88.9814749,0.488081123 87.9163292,0.795987208 87.1201941,1.49909291 C86.324059,2.20219861 85.8798887,3.22724309 85.9087059,4.29493319 L85.9087059,54.6039644 C85.9087059,56.9393437 87.6028236,57.8717845 89.5680001,56.7511446 L133.149176,31.9088684 C134.092985,31.4117209 134.762111,30.50855 134.968019,29.4538349 C135.173928,28.3991197 134.894328,27.3070442 134.208,26.4853136 C134.208,26.4853136 133.538824,25.5357637 132.446118,24.3381334 C127.296,18.6921614 122.832,14.7399812 116.860235,10.9417819 C109.345997,6.07770841 100.978781,2.71218494 92.2108235,1.0271132 C89.9745882,0.599388123 90.0254118,0.659269609 90.0254118,0.659269609 L90.0254118,0.659269609 Z" id="Shape"></path> </g> </g> </svg>'
            },
            702: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="151px" height="170px" viewBox="0 0 151 170" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>radar</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="radar" transform="translate(0.500000, 0.000000)" fill="#5067A2" fill-rule="nonzero"> <path d="M79.573911,32.0195055 C79.7511468,32.0807796 79.9252225,32.1507623 80.0954971,32.2291958 L126.376239,53.5077701 C128.225278,54.3572545 129.374745,56.2354539 129.285085,58.2607506 L126.927916,110.733256 C126.832633,112.851954 125.405699,114.680595 123.367088,115.296516 L76.444394,129.315811 C75.5054275,129.595861 74.5046062,129.595861 73.5656397,129.315811 L32.4706683,117.053921 C30.8367372,116.566659 29.5635821,115.286115 29.0909814,113.654618 C28.6183807,112.023122 29.0111288,110.264354 30.133561,108.985836 L59.2822032,75.635091 C59.7218624,75.142978 60.0532763,74.5647698 60.2551619,73.937598 L73.2246026,35.1848307 C74.1062345,32.570839 76.9431645,31.1565423 79.573911,32.0195055 L79.573911,32.0195055 Z M139.476076,121.926724 L75.0000016,158.991984 L10.5239274,121.926724 L10.5239274,47.816176 L75.0000016,10.7509165 L139.476076,47.816176 L139.476076,121.926724 Z M146.998953,40.6068232 L77.5076273,0.665810956 C75.9548023,-0.221936985 74.0452007,-0.221936985 72.4923757,0.665810956 L3.00105014,40.6068232 C1.44784731,41.4971353 0.491395456,43.1462367 0.493424274,44.9304378 L0.493424274,124.812462 C0.493424274,126.599823 1.44632208,128.247389 3.00105014,129.136077 L72.4923757,169.077089 C74.047355,169.95676 75.9526481,169.95676 77.5076273,169.077089 L146.998953,129.136077 C148.550638,128.244253 149.506533,126.596111 149.506579,124.812462 L149.506579,44.9304378 C149.506533,43.1467894 148.550638,41.4986473 146.998953,40.6068232 L146.998953,40.6068232 Z" id="Shape"></path> </g> </g> </svg>'
            },
            989: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="167px" height="162px" viewBox="0 0 167 162" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>rich</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="rich" fill="#5067A2" fill-rule="nonzero"> <path d="M5.52634288,101.24857 L96.0005147,101.24857 C98.7764873,101.24857 101.026858,98.9820436 101.026858,96.1861415 C101.026858,93.3902395 98.7764873,91.123713 96.0005147,91.123713 L5.52634288,91.123713 C2.75037036,91.123713 0.5,93.3902395 0.5,96.1861415 C0.5,98.9820436 2.75037036,101.24857 5.52634288,101.24857 Z M75.8951432,121.498284 L5.52634288,121.498284 C2.75037036,121.498284 0.5,123.76481 0.5,126.560713 C0.5,129.356615 2.75037036,131.623141 5.52634288,131.623141 L75.8951432,131.623141 C78.6711157,131.623141 80.9214861,129.356615 80.9214861,126.560713 C80.9214861,123.76481 78.6711157,121.498284 75.8951432,121.498284 Z M146.394628,92.0653247 L128.923061,116.435856 L146.394628,116.435856 L146.394628,92.0653247 Z M166.5,60.749142 L166.5,151.872855 C166.5,157.464659 161.999259,161.997712 156.447314,161.997712 C150.895369,161.997712 146.394628,157.464659 146.394628,151.872855 L146.394628,136.68557 L114.406982,136.68557 L99.2575849,157.806021 C96.0041451,162.338178 89.7188607,163.355849 85.2190093,160.079052 C80.7191578,156.802254 79.7087415,150.471846 82.9621813,145.939689 L148.304639,54.8261007 C148.455429,54.6033538 148.706746,54.4919804 148.867589,54.2894832 C149.318111,53.7697312 149.823755,53.3011987 150.375492,52.892253 C150.857637,52.4854402 151.379728,52.1292264 151.933658,51.829143 C152.481136,51.5631055 153.053631,51.3528145 153.642615,51.2014019 C154.31205,50.9842108 155.003114,50.8416105 155.703415,50.7761579 C155.954733,50.7559082 156.185944,50.624285 156.447314,50.624285 C156.819264,50.624285 157.130897,50.7964076 157.492794,50.836907 C158.850886,50.9717079 160.164137,51.3999382 161.342972,52.0923893 C161.674711,52.2746367 162.036608,52.3151361 162.338188,52.537883 C162.559347,52.7100056 162.659874,52.9428773 162.87098,53.1251247 C163.403773,53.5807433 163.836038,54.0869861 164.268304,54.6337284 C164.650306,55.1399712 165.022255,55.6259644 165.30373,56.1929564 C165.585206,56.7498235 165.766154,57.3370652 165.947102,57.9546815 C166.138103,58.6330469 166.298946,59.2709129 166.34921,59.969528 C166.369315,60.2428992 166.5,60.4757709 166.5,60.749142 L166.5,60.749142 Z M55.7897717,151.872855 L5.52634288,151.872855 C2.75037036,151.872855 0.5,154.139381 0.5,156.935284 C0.5,159.731186 2.75037036,161.997712 5.52634288,161.997712 L55.7897717,161.997712 C58.5657442,161.997712 60.8161146,159.731186 60.8161146,156.935284 C60.8161146,154.139381 58.5657442,151.872855 55.7897717,151.872855 Z M86.3887335,40.499428 L161.100608,40.499428 C164.04791,40.499428 166.437171,38.2329016 166.437171,35.4369995 C166.437171,32.6410974 164.04791,30.374571 161.100608,30.374571 L86.3887335,30.374571 C83.4414314,30.374571 81.052171,32.6410974 81.052171,35.4369995 C81.052171,38.2329016 83.4414314,40.499428 86.3887335,40.499428 Z M86.3887335,10.124857 L161.100608,10.124857 C164.04791,10.124857 166.437171,7.85833056 166.437171,5.0624285 C166.437171,2.26652644 164.04791,1.71199625e-16 161.100608,0 L86.3887335,0 C83.4414314,-1.71199625e-16 81.052171,2.26652644 81.052171,5.0624285 C81.052171,7.85833056 83.4414314,10.124857 86.3887335,10.124857 Z M5.6570278,70.873999 L60.9467995,70.873999 C63.722772,70.873999 65.9731424,68.6074726 65.9731424,65.8115705 L65.9731424,5.0624285 C65.9731424,2.26652644 63.722772,1.71199625e-16 60.9467995,0 L5.6570278,0 C2.88105528,-1.71199625e-16 0.630684915,2.26652644 0.630684915,5.0624285 L0.630684915,65.8115705 C0.630684915,68.6074726 2.88105528,70.873999 5.6570278,70.873999 Z M81.052171,65.8115705 C81.052171,63.0156685 83.3025414,60.749142 86.0785139,60.749142 L116.236571,60.749142 C119.012544,60.749142 121.262914,63.0156685 121.262914,65.8115705 C121.262914,68.6074726 119.012544,70.873999 116.236571,70.873999 L86.0785139,70.873999 C83.3025414,70.873999 81.052171,68.6074726 81.052171,65.8115705 Z" id="Shape"></path> </g> </g> </svg>'
            },
            827: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="139px" viewBox="0 0 175 139" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>sankey</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="sankey" fill-rule="nonzero"> <path d="M4.60195168,138.058552 C2.06036395,138.058552 3.11254728e-16,135.998188 0,133.4566 L0,115.048793 C-3.11254732e-16,112.507205 2.06036398,110.446841 4.60195174,110.446841 C7.14353951,110.446841 9.20390349,112.507205 9.20390349,115.048793 L9.20390349,133.4566 C9.20390349,134.677113 8.71905638,135.847638 7.856023,136.710671 C6.99298962,137.573705 5.82246517,138.058552 4.60195168,138.058552 M4.60195168,106.820503 C2.06036395,106.820503 3.11254728e-16,104.760139 0,102.218552 L0,4.60195168 C-3.11254732e-16,2.06036391 2.06036398,-6.74114049e-08 4.60195174,-6.74114054e-08 C7.14353951,-6.74114058e-08 9.20390349,2.06036391 9.20390349,4.60195168 L9.20390349,102.218552 C9.20390349,103.439065 8.71905638,104.60959 7.856023,105.472623 C6.99298962,106.335656 5.82246517,106.820503 4.60195168,106.820503" id="Shape" fill="#5067A2"></path> <path d="M4.60195168,106.820503 C2.06036395,106.820503 3.11254728e-16,104.760139 0,102.218552 L0,4.60195168 C-3.11254732e-16,2.06036391 2.06036398,-6.74114049e-08 4.60195174,-6.74114054e-08 C7.14353951,-6.74114058e-08 9.20390349,2.06036391 9.20390349,4.60195168 L9.20390349,102.218552 C9.20390349,103.439065 8.71905638,104.60959 7.856023,105.472623 C6.99298962,106.335656 5.82246517,106.820503 4.60195168,106.820503" id="Path" fill="#5067A2"></path> <path d="M87.4370826,64.4273241 C86.2165692,64.4273241 85.0460447,63.942477 84.1830114,63.0794436 C83.319978,62.2164103 82.8351309,61.0458858 82.835131,59.8253723 L82.835131,4.60195168 C82.835131,2.06036391 84.8954949,-6.74113054e-08 87.4370827,-6.74113059e-08 C89.9786705,-6.74113063e-08 92.0390345,2.06036391 92.0390345,4.60195168 L92.0390345,59.8253723 C92.0390345,61.0458858 91.5541874,62.2164103 90.691154,63.0794437 C89.8281206,63.9424771 88.6575962,64.4273242 87.4370826,64.4273241 M170.272214,38.2054031 C167.730626,38.2054031 165.670262,36.1450392 165.670262,33.6034515 L165.670262,4.60195168 C165.670262,2.06036391 167.730626,-6.74114049e-08 170.272214,-6.74114054e-08 C172.813801,-6.74114058e-08 174.874165,2.06036391 174.874165,4.60195168 L174.874165,33.6034515 C174.874165,34.823965 174.389318,35.9944894 173.526285,36.8575227 C172.663252,37.7205561 171.492727,38.2054032 170.272214,38.2054031 M105.84489,138.058552 C104.624376,138.058552 103.453852,137.573705 102.590818,136.710671 C101.727785,135.847638 101.242938,134.677113 101.242938,133.4566 L101.242938,78.2331793 C101.242938,75.6915915 103.303302,73.6312276 105.84489,73.6312276 C108.386477,73.6312276 110.446841,75.6915915 110.446841,78.2331793 L110.446841,133.4566 C110.446841,135.998188 108.386477,138.058552 105.84489,138.058552 M170.272214,138.058552 C167.730626,138.058552 165.670262,135.998188 165.670262,133.4566 L165.670262,48.3020853 C165.670262,45.7604975 167.730626,43.7001336 170.272214,43.7001336 C172.813801,43.7001336 174.874165,45.7604975 174.874165,48.3020853 L174.874165,133.4566 C174.874165,134.677113 174.389318,135.847638 173.526285,136.710671 C172.663252,137.573705 171.492727,138.058552 170.272214,138.058552" id="Shape" fill="#5067A2"></path> <polygon id="Path" fill="#95A5C8" points="110.446841 133.4566 165.670262 133.4566 165.670262 78.2331793 110.446841 78.2331793"></polygon> <polygon id="Path" fill="#5067A2" points="165.670262 133.4566 174.874165 133.4566 174.874165 78.2331793 165.670262 78.2331793"></polygon> <path d="M165.670262,77.9478582 C124.491998,75.6100668 130.198418,59.8345763 92.0390345,59.8253723 L92.0390345,32.213662 C124.832542,32.2228659 136.825229,48.0167643 165.670262,48.6242219 L165.670262,77.9478582" id="Path" fill="#95A5C8"></path> <path d="M92.0390345,59.8253723 L92.0022188,59.8253723 L92.0022188,32.213662 L92.0390345,32.213662 L92.0390345,59.8253723 M174.874165,78.2331793 C171.56076,78.2055676 168.505064,78.1135285 165.670262,77.9478582 L165.670262,48.6242219 L167.400596,48.6426296 C169.765999,48.6426296 172.260257,48.5413867 174.874165,48.3204931 L174.874165,78.2331793" id="Shape" fill="#5067A2"></path> <polygon id="Path" fill="#95A5C8" points="92.0390345 32.213662 165.670262 32.213662 165.670262 4.60195168 92.0390345 4.60195168"></polygon> <path d="M82.835131,32.213662 L92.0390345,32.213662 L92.0390345,4.60195168 L82.835131,4.60195168 L82.835131,32.213662 Z M165.670262,32.213662 L174.506009,32.213662 L174.506009,4.60195168 L165.670262,4.60195168 L165.670262,32.213662 Z" id="Shape" fill="#5067A2"></path> <polygon id="Path" fill="#95A5C8" points="9.20390349 59.8253723 82.835131 59.8253723 82.835131 4.60195168 9.20390349 4.60195168"></polygon> <polyline id="Path" fill="#5067A2" points="82.835131 59.8253723 82.835131 4.60195168 82.835131 59.8253723"></polyline> <polygon id="Path" fill="#95A5C8" points="9.20390349 133.4566 101.242938 133.4566 101.242938 115.048793 9.20390349 115.048793"></polygon> <polygon id="Path" fill="#5067A2" points="101.242938 133.4566 110.446841 133.4566 110.446841 115.048793 101.242938 115.048793"></polygon> <path d="M101.242938,115.048793 C43.4792398,114.699045 52.1585208,101.445424 9.20390349,101.445424 L9.20390349,59.8253723 C56.5579867,59.8253723 47.7682589,78.1319363 101.242938,78.2331793 L101.242938,115.048793" id="Path" fill="#95A5C8"></path> <path d="M101.675521,115.048793 L101.242938,115.048793 L101.242938,78.2331793 L101.675521,78.2331793 C104.335449,78.2331793 100.46981,113.723431 101.675521,115.048793" id="Path" fill="#5067A2"></path> </g> </g> </svg>'
            },
            687: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="138px" viewBox="0 0 175 138" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>scatter</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="scatter" fill="#5067A2" fill-rule="nonzero"> <path d="M173.526285,129.202529 C172.663252,128.339495 171.492727,127.854648 170.272214,127.854648 L9.20390349,127.854648 L9.20390349,4.60195168 C9.20390349,2.06036391 7.14353951,-6.74112924e-08 4.60195174,-6.74112925e-08 C2.06036398,-6.74112927e-08 3.11254732e-16,2.06036391 0,4.60195168 L0,132.4566 C3.11254728e-16,134.998188 2.06036395,137.058552 4.60195168,137.058552 L170.272214,137.058552 C171.492727,137.058552 172.663252,136.573705 173.526285,135.710671 C174.389318,134.847638 174.874165,133.677113 174.874165,132.4566 C174.874165,131.236086 174.389318,130.065562 173.526285,129.202529 Z M151.864407,12.8058552 C155.15291,12.8054572 158.191783,14.5596242 159.836149,17.4074853 C161.480516,20.2553464 161.480516,23.7641708 159.836149,26.6120319 C158.191783,29.459893 155.15291,31.21406 151.864407,31.213662 C146.781666,31.2130468 142.661617,27.0924991 142.661617,22.0097586 C142.661617,16.9270181 146.781666,12.8064703 151.864407,12.8058552 Z M142.660503,58.8253723 C147.743679,58.8253723 151.864407,62.9461003 151.864407,68.0292758 C151.864407,73.1124513 147.743679,77.2331793 142.660503,77.2331793 C137.577328,77.2331793 133.4566,73.1124513 133.4566,68.0292758 C133.4566,62.9461003 137.577328,58.8253723 142.660503,58.8253723 Z M124.252696,40.4175655 C127.5412,40.4171675 130.580072,42.1713345 132.224439,45.0191956 C133.868806,47.8670567 133.868806,51.3758811 132.224439,54.2237422 C130.580072,57.0716033 127.5412,58.8257703 124.252696,58.8253723 C119.169956,58.8247572 115.049907,54.7042094 115.049907,49.6214689 C115.049907,44.5387284 119.169956,40.4181807 124.252696,40.4175655 L124.252696,40.4175655 Z M105.84489,22.0097586 C109.133393,22.0093606 112.172266,23.7635277 113.816632,26.6113888 C115.460999,29.4592498 115.460999,32.9680743 113.816632,35.8159354 C112.172266,38.6637965 109.133393,40.4179635 105.84489,40.4175655 C100.762149,40.4169503 96.6421001,36.2964026 96.6421001,31.2136621 C96.6421001,26.1309216 100.762149,22.0103738 105.84489,22.0097586 L105.84489,22.0097586 Z M105.84489,58.8253723 C110.928065,58.8253723 115.048793,62.9461003 115.048793,68.0292758 C115.048793,73.1124513 110.928065,77.2331793 105.84489,77.2331793 C100.761714,77.2331793 96.6409861,73.1124513 96.6409861,68.0292758 C96.6409861,62.9461003 100.761714,58.8253723 105.84489,58.8253723 L105.84489,58.8253723 Z M87.4370826,77.2331793 C90.8200008,77.0784213 94.0146452,78.7938663 95.7543045,81.6993216 C97.4939639,84.6047768 97.4975937,88.2308616 95.7637545,91.1397939 C94.0299154,94.0487262 90.8387117,95.7705634 87.4554905,95.6225784 C82.5370015,95.4074394 78.6587219,91.3598697 78.6537937,86.4366802 C78.6488656,81.5134907 82.5190342,77.4581647 87.4370826,77.2331793 L87.4370826,77.2331793 Z M59.8253723,68.0292758 C63.1138756,68.0288778 66.1527483,69.7830448 67.7971148,72.6309059 C69.4414814,75.478767 69.4414814,78.9875915 67.7971148,81.8354525 C66.1527483,84.6833136 63.1138756,86.4374807 59.8253723,86.4370826 C54.7426318,86.4364675 50.6225828,82.3159197 50.6225828,77.2331792 C50.6225828,72.1504387 54.7426318,68.029891 59.8253723,68.0292758 L59.8253723,68.0292758 Z M32.213662,95.6409861 C35.5021653,95.6405881 38.541038,97.3947552 40.1854045,100.242616 C41.8297711,103.090477 41.8297711,106.599302 40.1854045,109.447163 C38.541038,112.295024 35.5021653,114.049191 32.213662,114.048793 C27.1309215,114.048178 23.0108725,109.92763 23.0108725,104.84489 C23.0108725,99.762149 27.1309215,95.6416013 32.213662,95.6409861 L32.213662,95.6409861 Z" id="Shape"></path> </g> </g> </svg>'
            },
            31: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="176px" height="154px" viewBox="0 0 176 154" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>sunburst</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="sunburst" transform="translate(-0.500000, -0.495167)" fill="#5067A2" fill-rule="nonzero"> <path d="M119.233629,98.0044542 L110.718691,87.8510962 C116.698944,81.6134116 119.776927,73.1460671 119.198407,64.5239101 L132.221771,61.7588152 C132.389075,63.3350954 132.468325,64.9289877 132.468325,66.5404921 C132.486789,78.3855449 127.713147,89.7343283 119.233629,98.0044542 L119.233629,98.0044542 Z M63.3185781,30.3829135 C74.7931609,22.4121709 89.3686325,20.3583561 102.598787,24.8479794 C115.828942,29.3376027 126.144611,39.8381699 130.399028,53.1465133 L117.38447,55.9116082 C114.163414,47.1482037 107.14477,40.3166874 98.2978797,37.3338959 C89.4509892,34.3511045 79.7290566,35.5384174 71.8599325,40.5626897 L63.3185781,30.3829135 L63.3185781,30.3829135 Z M57.4981296,97.8547516 L67.6773104,89.3128978 C73.1543438,94.3147256 80.4453141,97.3616135 88.4495316,97.3616137 C93.8284022,97.3676833 99.1146581,95.9617789 103.779942,93.2844195 L112.365323,103.508226 C105.247152,108.132446 96.9377023,110.586264 88.4495316,110.570666 C76.8578752,110.587037 65.7306915,106.015602 57.4981296,97.8547516 L57.4981296,97.8547516 Z M51.880208,91.0741048 C47.0035405,83.8229191 44.4062144,75.2791485 44.4219328,66.5404921 C44.4219328,54.7315996 49.0624416,44.0146553 56.6263832,36.1068361 L65.194154,46.3130303 C60.3076881,51.9176198 57.6201378,59.1046557 57.6302124,66.5404921 C57.6302124,72.3876992 59.2592335,77.8562467 62.0858054,82.5146391 L51.880208,91.0741048 L51.880208,91.0741048 Z M40.5915316,112.05008 L50.7354904,103.534644 C60.658337,113.682313 74.2570793,119.394563 88.4495316,119.3767 C99.4212094,119.3767 109.618001,116.030407 118.062495,110.306485 L126.586238,120.468649 C115.441046,128.370969 102.111678,132.606084 88.4495316,132.585752 C70.3533434,132.605952 53.0459635,125.179425 40.5915316,112.05008 L40.5915316,112.05008 Z M0.5,62.1374747 C1.61718015,39.3368661 11.5653323,17.8665508 28.2373873,2.27405071 L36.7435195,12.4097967 C23.0478109,25.4703646 14.8209875,43.2437342 13.7258907,62.1374747 L0.5,62.1374747 Z M0.5,70.9435094 L13.7258907,70.9435094 C14.5580363,85.4042184 19.5854188,99.3091254 28.1933598,110.958131 L18.0582064,119.464761 C7.47186066,105.41813 1.35420505,88.5122858 0.5,70.9435094 L0.5,70.9435094 Z M22.5490216,62.1374747 C23.6176776,45.8469855 30.7006658,30.5346593 42.4230797,19.1728312 L50.9380174,29.3261893 C42.1806215,38.1364527 36.8165849,49.7571326 35.7925234,62.1374747 L22.5490216,62.1374747 L22.5490216,62.1374747 Z M22.5490216,70.9435094 L35.7925234,70.9435094 C36.5571371,80.2216635 39.7688087,89.1310032 45.0999577,96.7628032 L34.9559989,105.278239 C27.6604148,95.2350677 23.3587128,83.3306553 22.5490216,70.9435094 Z M160.064824,88.3354281 C162.208458,81.2692542 163.294499,73.9246962 163.287644,66.5404921 C163.287927,62.7888436 163.014228,59.0421838 162.468731,55.3304099 L175.40404,52.5829271 C177.539642,66.0542744 176.545357,79.8364744 172.498218,92.86173 L160.064824,88.3354281 Z M157.009309,96.5954887 L169.451508,101.130596 C164.453629,112.804913 156.992417,123.261434 147.578597,131.784403 L139.08127,121.666269 C146.731696,114.64362 152.837128,106.105711 157.009309,96.5954887 L157.009309,96.5954887 Z M139.336631,80.7886563 C141.234304,74.0037443 141.75525,66.9076826 140.868791,59.9183539 L153.830516,57.1620651 C155.148964,66.5886629 154.449628,76.187832 151.77883,85.3237642 L139.336631,80.7886563 Z M136.263504,89.0487169 L148.714509,93.5750188 C145.090489,101.633987 139.889912,108.885878 133.419321,114.903235 L124.913189,104.776295 C129.629614,100.274719 133.477161,94.943361 136.263504,89.0487169 L136.263504,89.0487169 Z M84.1260215,141.268503 L84.1260215,154.495167 C61.049526,153.386197 39.3397973,143.226766 23.7025446,126.218989 L33.8288923,117.71236 C46.9380429,131.745712 64.9535467,140.183088 84.1260215,141.268503 L84.1260215,141.268503 Z M104.94227,2.57345585 C127.679056,8.4576749 145.591926,25.956003 152.007773,48.5497632 L139.046048,51.306052 C133.813709,33.9700374 120.081572,20.5128194 102.64403,15.6328053 L104.933465,2.64390412 L104.94227,2.57345585 L104.94227,2.57345585 Z M96.2952498,0.953145396 L96.2600278,1.12046007 L93.9970091,13.9948829 C81.1068487,12.6102912 68.1594065,16.034712 57.6390179,23.6110727 L49.1240802,13.4665207 C60.4957814,5.02434618 74.2870046,0.475385223 88.4495316,0.495231624 C91.0999932,0.495231624 93.7152325,0.644934198 96.2952498,0.953145396 L96.2952498,0.953145396 Z" id="Shape"></path> <path d="M88.5,75.295167 C93.3601058,75.295167 97.3,71.3552728 97.3,66.495167 C97.3,61.6350611 93.3601058,57.6951669 88.5,57.6951669 C83.6398942,57.6951669 79.7,61.6350611 79.7,66.495167 C79.7,71.3552728 83.6398942,75.295167 88.5,75.295167 Z M88.5,88.495167 C76.3497355,88.495167 66.5,78.6454315 66.5,66.495167 C66.5,54.3449025 76.3497355,44.495167 88.5,44.495167 C100.650264,44.495167 110.5,54.3449025 110.5,66.495167 C110.5,78.6454315 100.650264,88.495167 88.5,88.495167 Z" id="Shape"></path> </g> </g> </svg>'
            },
            951: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="175px" height="130px" viewBox="0 0 175 130" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>themeriver</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="themeriver" fill="#5067A2" fill-rule="nonzero"> <path d="M4.9275627,42.5798026 C5.20117275,42.5813119 5.47437729,42.5585448 5.74396361,42.5117692 C10.5811329,41.6686599 15.2612908,40.0922201 19.6227791,37.836902 C26.4649962,34.328322 32.1798026,30.0130599 43.8718299,30.0130599 C55.5638572,30.0130599 63.7861807,41.3940774 79.5602127,41.4912681 C95.3342445,41.5787397 96.9864845,31.7333333 115.384662,31.7333333 C132.976158,31.7333333 137.621868,44.6208049 153.6,45.4566438 C160.490812,45.8162491 166.6041,44.9318147 171.930144,42.7936218 C173.778744,42.0602993 174.992296,40.2727177 174.991648,38.2839788 L174.991648,26.7960517 C174.987486,25.3270305 174.319017,23.9387356 173.173189,23.0194311 C172.027362,22.1001266 170.527173,21.7484925 169.092179,22.0628701 C165.933485,22.7529233 162.084738,22.8403949 157.5265,22.3155656 C145.533181,20.9160213 134.676993,5.82232335 119.476386,5.82232348 C102.788762,5.82232348 99.3093394,11.1678056 81.9899772,11.1678056 C64.6803341,11.1678056 59.6069856,0.126955258 43.8718299,0.126955258 C37.3211845,0.126955258 27.9422931,3.46059231 17.9025057,7.6009112 C12.9652241,9.65163256 8.49445708,10.8373576 4.50964317,11.1678056 C1.98627058,11.3754189 0.0451326562,13.4857192 0.0485952544,16.0176158 L0.0583143318,37.7202733 C0.0583143318,40.4027335 2.24510251,42.5798026 4.9275627,42.5798026 M170.132118,85.5283219 C169.634444,85.5282959 169.139638,85.6036638 168.664541,85.7518603 C162.658162,87.6665148 157.633409,88.7064541 153.551405,88.9008353 C139.342141,89.522855 131.343356,82.7583903 118.543356,82.7583903 C104.227183,82.7583903 97.8611998,94.1491269 80.444647,94.1491269 C61.0356872,94.1491269 52.3274108,84.1287775 42.4917236,84.1287775 C35.416249,84.1287775 28.7392558,88.7453303 19.9337889,92.9828398 C15.5893698,95.0627183 10.3507973,96.5594533 4.2083523,97.4438876 C1.81553905,97.7920745 0.0427391378,99.8465342 0.0485952544,102.264541 L0.0485952544,124.297646 C0.0496707163,125.687409 0.645723287,127.010209 1.68608385,127.931672 C2.72644442,128.853134 4.11154706,129.285071 5.491268,129.118299 C10.3313591,128.544875 15.142293,127.670159 19.9337889,126.513592 C29.8083523,124.132422 34.2305239,119.068793 42.4917236,118.388459 C54.5822323,117.397115 66.5658314,123.267426 82.4759302,121.323614 C98.386029,119.379803 106.3265,110.156416 120.273349,108.543052 C132.577676,107.12407 140.586181,111.293546 152.083827,110.030068 C159.995141,109.174791 166.81792,107.085194 172.552164,103.780714 C174.06095,102.911932 174.991067,101.303681 174.991648,99.5626424 L174.991648,90.3878512 C174.991648,87.7040073 172.815962,85.5283219 170.132118,85.5283219" id="Shape"></path> <path d="M170.132118,55.1659834 C169.755896,55.1642513 169.380769,55.206657 169.014427,55.2923311 C163.386911,56.5597226 157.597804,56.9530934 151.850569,56.4586181 C133.889749,55.0007593 128.388762,42.7839029 114.655733,42.7839029 C103.264996,42.7839029 93.9249811,52.940319 79.7157175,52.5807138 C59.2668185,52.0656038 52.6578588,41.0733486 42.0738042,41.0733486 C35.5231587,41.0733486 26.5330296,47.0991647 20.6724373,49.7524677 C15.5296251,52.0293301 10.0450596,53.4382394 4.44160976,53.9219438 C1.94888222,54.1623556 0.0470459996,56.2577415 0.0485952544,58.762035 L0.0485952544,80.8242976 C0.0491745866,82.2971429 0.717699925,83.69025 1.86632297,84.6121711 C3.01494602,85.5340923 4.51970635,85.8853329 5.95778278,85.5671982 C11.10094,84.3721963 16.0531556,82.4685044 20.6724373,79.9107062 C28.3018983,75.7703873 34.249962,72.7866364 41.8016705,72.7866364 C56.6037965,72.7866364 63.1252849,82.7778284 79.3755505,82.7778284 C94.6441914,82.7778284 103.070615,71.5231587 117.639484,71.5231587 C132.208352,71.5231587 139.964161,77.2185269 150.314958,77.5295368 C156.48656,77.7044798 163.707821,76.0813971 171.988459,72.6408505 C173.80584,71.8896408 174.991328,70.1171628 174.991648,68.1506454 L174.991648,60.0255125 C174.991648,57.3416686 172.815962,55.1659834 170.132118,55.1659834" id="Path"></path> </g> </g> </svg>'
            },
            929: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="174px" height="147px" viewBox="0 0 174 147" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>tree (1)</title> <desc>Created with Sketch.</desc> <defs> <polygon id="path-1" points="2 4.69629167 172 4.69629167 172 146.375 2 146.375"></polygon> </defs> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="tree-(1)"> <mask id="mask-2" fill="white"> <use xlink:href="#path-1"></use> </mask> <g id="path-1"></g> <path d="M174,127.4965 C173.994333,137.928125 165.534,146.380667 155.101667,146.375708 C144.670042,146.370042 136.216792,137.909 136.222458,127.476667 C136.222458,127.470292 136.222458,127.464625 136.222458,127.45825 C136.226708,118.755667 142.117917,111.435042 150.127042,109.248417 L150.127042,80.2563333 L91.4625,80.2563333 L91.4625,109.120208 C99.7407917,111.126208 105.890542,118.582125 105.889125,127.476667 C105.889125,127.483042 105.889125,127.490125 105.889125,127.4965 C105.883458,137.928125 97.4224167,146.380667 86.9907917,146.375708 C76.5584583,146.370042 68.1059167,137.909 68.1115833,127.476667 C68.1122917,118.764875 74.0049167,111.440708 82.0182917,109.254083 L82.0182917,80.2563333 L23.3530417,80.2563333 L23.3530417,109.120917 C31.630625,111.127625 37.7789583,118.582833 37.77825,127.476667 C37.77825,127.478792 37.77825,127.480917 37.77825,127.483042 C37.776125,137.915375 29.3179167,146.37075 18.8862917,146.368625 C8.45395833,146.367208 -0.00141666667,137.909 0,127.476667 C0.00141666667,118.764167 5.89545833,111.439292 13.9088333,109.254083 L13.9088333,80.2563333 C13.9088333,75.0500833 18.1425417,70.812125 23.3530417,70.812125 L82.0182917,70.812125 L82.0182917,37.8179167 C74.0006667,35.6305833 68.106625,28.298625 68.1115833,19.58825 C68.1122917,9.15025 76.5705,0.694875 87.0028333,0.696291667 C97.4344583,0.697708333 105.890542,9.15591667 105.889125,19.58825 C105.889125,19.594625 105.889125,19.601 105.889125,19.607375 C105.884167,28.4976667 99.7358333,35.9465 91.4625,37.9517917 L91.4625,70.812125 L150.127042,70.812125 C155.337542,70.812125 159.57125,75.0500833 159.57125,80.2563333 L159.57125,109.113125 C167.853083,111.119125 174.00425,118.579292 174,127.476667 C174,127.483042 174,127.490125 174,127.4965 Z" id="Fill-1" fill="#5067A2" fill-rule="nonzero"></path> </g> </g> </svg>'
            },
            101: t => {
                t.exports = '<?xml version="1.0" encoding="UTF-8"?> <svg width="160px" height="132px" viewBox="0 0 160 132" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <title>treemap</title> <desc>Created with Sketch.</desc> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="treemap" transform="translate(-0.500000, -0.095199)" fill="#5067A2" fill-rule="nonzero"> <path d="M84.9444444,79.2951993 L4.9444444,79.2951993 C2.48984554,79.2951993 0.5,81.2651464 0.5,83.6951993 L0.5,127.695199 C0.5,130.125252 2.48984554,132.095199 4.9444444,132.095199 L84.9444444,132.095199 C86.1231844,132.095199 87.2536462,131.631629 88.0871413,130.806469 C88.9206363,129.981309 89.3888889,128.862152 89.3888889,127.695199 L89.3888889,83.6951993 C89.3888889,82.5282467 88.9206363,81.4090895 88.0871413,80.5839295 C87.2536462,79.7587694 86.1231844,79.2951993 84.9444444,79.2951993 L84.9444444,79.2951993 Z M156.055556,105.695199 L102.722222,105.695199 C101.543482,105.695199 100.41302,106.158769 99.5795254,106.983929 C98.7460304,107.80909 98.2777777,108.928247 98.2777777,110.095199 L98.2777777,127.695199 C98.2777777,128.862152 98.7460304,129.981309 99.5795254,130.806469 C100.41302,131.631629 101.543482,132.095199 102.722222,132.095199 L156.055556,132.095199 C158.510154,132.095199 160.5,130.125252 160.5,127.695199 L160.5,110.095199 C160.5,107.665146 158.510154,105.695199 156.055556,105.695199 L156.055556,105.695199 Z M156.055556,61.6951994 L102.722222,61.6951994 C101.543482,61.6951994 100.41302,62.1587694 99.5795254,62.9839295 C98.7460304,63.8090896 98.2777777,64.9282468 98.2777777,66.0951993 L98.2777777,92.4951993 C98.2777777,93.6621519 98.7460304,94.7813091 99.5795254,95.6064692 C100.41302,96.4316293 101.543482,96.8951993 102.722222,96.8951993 L156.055556,96.8951993 C158.510154,96.8951993 160.5,94.9252522 160.5,92.4951993 L160.5,66.0951993 C160.5,63.6651465 158.510154,61.6951994 156.055556,61.6951994 L156.055556,61.6951994 Z M156.055556,0.0951993491 L102.722222,0.0951993491 C101.543482,0.0951993149 100.41302,0.558769393 99.5795254,1.38392945 C98.7460304,2.20908951 98.2777777,3.32824674 98.2777777,4.49519931 L98.2777777,48.4951994 C98.2777777,49.662152 98.7460304,50.7813092 99.5795254,51.6064692 C100.41302,52.4316293 101.543482,52.8951994 102.722222,52.8951993 L156.055556,52.8951993 C158.510154,52.8951993 160.5,50.9252523 160.5,48.4951994 L160.5,4.49519931 C160.5,2.06514643 158.510154,0.0951993491 156.055556,0.0951993491 L156.055556,0.0951993491 Z M89.3888889,4.49519931 L89.3888889,66.0951993 C89.3888889,67.2621519 88.9206363,68.3813091 88.0871413,69.2064692 C87.2536462,70.0316293 86.1231844,70.4951993 84.9444444,70.4951993 L4.9444444,70.4951993 C2.48984554,70.4951993 0.5,68.5252522 0.5,66.0951993 L0.5,4.49519931 C0.5,2.06514643 2.48984554,0.0951993491 4.9444444,0.0951993491 L84.9444444,0.0951993491 C86.1231844,0.0951993149 87.2536462,0.558769393 88.0871413,1.38392945 C88.9206363,2.20908951 89.3888889,3.32824674 89.3888889,4.49519931 Z" id="Shape"></path> </g> </g> </svg>'
            },
            705: (t, e, n) => {
                var a = n(639).Symbol;
                t.exports = a
            },
            239: (t, e, n) => {
                var a = n(705),
                    i = n(607),
                    o = n(333),
                    r = a ? a.toStringTag : void 0;
                t.exports = function(t) {
                    return null == t ? void 0 === t ? "[object Undefined]" : "[object Null]" : r && r in Object(t) ? i(t) : o(t)
                }
            },
            957: (t, e, n) => {
                var a = "object" == typeof n.g && n.g && n.g.Object === Object && n.g;
                t.exports = a
            },
            607: (t, e, n) => {
                var a = n(705),
                    i = Object.prototype,
                    o = i.hasOwnProperty,
                    r = i.toString,
                    l = a ? a.toStringTag : void 0;
                t.exports = function(t) {
                    var e = o.call(t, l),
                        n = t[l];
                    try {
                        t[l] = void 0;
                        var a = !0
                    } catch (t) {}
                    var i = r.call(t);
                    return a && (e ? t[l] = n : delete t[l]), i
                }
            },
            333: t => {
                var e = Object.prototype.toString;
                t.exports = function(t) {
                    return e.call(t)
                }
            },
            639: (t, e, n) => {
                var a = n(957),
                    i = "object" == typeof self && self && self.Object === Object && self,
                    o = a || i || Function("return this")();
                t.exports = o
            },
            279: (t, e, n) => {
                var a = n(218),
                    i = n(771),
                    o = n(841),
                    r = Math.max,
                    l = Math.min;
                t.exports = function(t, e, n) {
                    var s, c, u, d, p, f, g = 0,
                        m = !1,
                        h = !1,
                        C = !0;
                    if ("function" != typeof t) throw new TypeError("Expected a function");

                    function y(e) {
                        var n = s,
                            a = c;
                        return s = c = void 0, g = e, d = t.apply(a, n)
                    }

                    function v(t) {
                        return g = t, p = setTimeout(_, e), m ? y(t) : d
                    }

                    function b(t) {
                        var n = t - f;
                        return void 0 === f || n >= e || n < 0 || h && t - g >= u
                    }

                    function _() {
                        var t = i();
                        if (b(t)) return L(t);
                        p = setTimeout(_, function(t) {
                            var n = e - (t - f);
                            return h ? l(n, u - (t - g)) : n
                        }(t))
                    }

                    function L(t) {
                        return p = void 0, C && s ? y(t) : (s = c = void 0, d)
                    }

                    function w() {
                        var t = i(),
                            n = b(t);
                        if (s = arguments, c = this, f = t, n) {
                            if (void 0 === p) return v(f);
                            if (h) return clearTimeout(p), p = setTimeout(_, e), y(f)
                        }
                        return void 0 === p && (p = setTimeout(_, e)), d
                    }
                    return e = o(e) || 0, a(n) && (m = !!n.leading, u = (h = "maxWait" in n) ? r(o(n.maxWait) || 0, e) : u, C = "trailing" in n ? !!n.trailing : C), w.cancel = function() {
                        void 0 !== p && clearTimeout(p), g = 0, s = f = c = p = void 0
                    }, w.flush = function() {
                        return void 0 === p ? d : L(i())
                    }, w
                }
            },
            218: t => {
                t.exports = function(t) {
                    var e = typeof t;
                    return null != t && ("object" == e || "function" == e)
                }
            },
            5: t => {
                t.exports = function(t) {
                    return null != t && "object" == typeof t
                }
            },
            448: (t, e, n) => {
                var a = n(239),
                    i = n(5);
                t.exports = function(t) {
                    return "symbol" == typeof t || i(t) && "[object Symbol]" == a(t)
                }
            },
            771: (t, e, n) => {
                var a = n(639);
                t.exports = function() {
                    return a.Date.now()
                }
            },
            841: (t, e, n) => {
                var a = n(218),
                    i = n(448),
                    o = /^\s+|\s+$/g,
                    r = /^[-+]0x[0-9a-f]+$/i,
                    l = /^0b[01]+$/i,
                    s = /^0o[0-7]+$/i,
                    c = parseInt;
                t.exports = function(t) {
                    if ("number" == typeof t) return t;
                    if (i(t)) return NaN;
                    if (a(t)) {
                        var e = "function" == typeof t.valueOf ? t.valueOf() : t;
                        t = a(e) ? e + "" : e
                    }
                    if ("string" != typeof t) return 0 === t ? t : +t;
                    t = t.replace(o, "");
                    var n = l.test(t);
                    return n || s.test(t) ? c(t.slice(2), n ? 2 : 8) : r.test(t) ? NaN : +t
                }
            },
            463: function(t) {
                "undefined" != typeof self && self, t.exports = function(t) {
                    var e = {};

                    function n(a) {
                        if (e[a]) return e[a].exports;
                        var i = e[a] = {
                            i: a,
                            l: !1,
                            exports: {}
                        };
                        return t[a].call(i.exports, i, i.exports, n), i.l = !0, i.exports
                    }
                    return n.m = t, n.c = e, n.d = function(t, e, a) {
                        n.o(t, e) || Object.defineProperty(t, e, {
                            enumerable: !0,
                            get: a
                        })
                    }, n.r = function(t) {
                        "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(t, Symbol.toStringTag, {
                            value: "Module"
                        }), Object.defineProperty(t, "__esModule", {
                            value: !0
                        })
                    }, n.t = function(t, e) {
                        if (1 & e && (t = n(t)), 8 & e) return t;
                        if (4 & e && "object" == typeof t && t && t.__esModule) return t;
                        var a = Object.create(null);
                        if (n.r(a), Object.defineProperty(a, "default", {
                                enumerable: !0,
                                value: t
                            }), 2 & e && "string" != typeof t)
                            for (var i in t) n.d(a, i, function(e) {
                                return t[e]
                            }.bind(null, i));
                        return a
                    }, n.n = function(t) {
                        var e = t && t.__esModule ? function() {
                            return t.default
                        } : function() {
                            return t
                        };
                        return n.d(e, "a", e), e
                    }, n.o = function(t, e) {
                        return Object.prototype.hasOwnProperty.call(t, e)
                    }, n.p = "/dist/", n(n.s = 1)
                }([function(t, e) {
                    var n = "function" == typeof Float32Array;

                    function a(t, e) {
                        return 1 - 3 * e + 3 * t
                    }

                    function i(t, e) {
                        return 3 * e - 6 * t
                    }

                    function o(t) {
                        return 3 * t
                    }

                    function r(t, e, n) {
                        return ((a(e, n) * t + i(e, n)) * t + o(e)) * t
                    }

                    function l(t, e, n) {
                        return 3 * a(e, n) * t * t + 2 * i(e, n) * t + o(e)
                    }

                    function s(t) {
                        return t
                    }
                    t.exports = function(t, e, a, i) {
                        if (!(0 <= t && t <= 1 && 0 <= a && a <= 1)) throw new Error("bezier x values must be in [0, 1] range");
                        if (t === e && a === i) return s;
                        for (var o = n ? new Float32Array(11) : new Array(11), c = 0; c < 11; ++c) o[c] = r(.1 * c, t, a);

                        function u(e) {
                            for (var n = 0, i = 1; 10 !== i && o[i] <= e; ++i) n += .1;
                            --i;
                            var s = n + (e - o[i]) / (o[i + 1] - o[i]) * .1,
                                c = l(s, t, a);
                            return c >= .001 ? function(t, e, n, a) {
                                for (var i = 0; i < 4; ++i) {
                                    var o = l(e, n, a);
                                    if (0 === o) return e;
                                    e -= (r(e, n, a) - t) / o
                                }
                                return e
                            }(e, s, t, a) : 0 === c ? s : function(t, e, n, a, i) {
                                var o, l, s = 0;
                                do {
                                    (o = r(l = e + (n - e) / 2, a, i) - t) > 0 ? n = l : e = l
                                } while (Math.abs(o) > 1e-7 && ++s < 10);
                                return l
                            }(e, n, n + .1, t, a)
                        }
                        return function(t) {
                            return 0 === t ? 0 : 1 === t ? 1 : r(u(t), e, i)
                        }
                    }
                }, function(t, e, n) {
                    "use strict";
                    n.r(e);
                    var a = function() {
                        var t = this.$createElement;
                        return (this._self._c || t)(this.tag, {
                            ref: "scrollactive-nav-wrapper",
                            tag: "component",
                            staticClass: "scrollactive-nav"
                        }, [this._t("default")], 2)
                    };
                    a._withStripped = !0;
                    var i = n(0),
                        o = n.n(i);

                    function r(t) {
                        return function(t) {
                            if (Array.isArray(t)) return l(t)
                        }(t) || function(t) {
                            if ("undefined" != typeof Symbol && Symbol.iterator in Object(t)) return Array.from(t)
                        }(t) || function(t, e) {
                            if (t) {
                                if ("string" == typeof t) return l(t, e);
                                var n = Object.prototype.toString.call(t).slice(8, -1);
                                return "Object" === n && t.constructor && (n = t.constructor.name), "Map" === n || "Set" === n ? Array.from(t) : "Arguments" === n || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? l(t, e) : void 0
                            }
                        }(t) || function() {
                            throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")
                        }()
                    }

                    function l(t, e) {
                        (null == e || e > t.length) && (e = t.length);
                        for (var n = 0, a = new Array(e); n < e; n++) a[n] = t[n];
                        return a
                    }
                    var s = function(t, e, n, a, i, o, r, l) {
                        var s, c = "function" == typeof t ? t.options : t;
                        if (e && (c.render = e, c.staticRenderFns = [], c._compiled = !0), s)
                            if (c.functional) {
                                c._injectStyles = s;
                                var u = c.render;
                                c.render = function(t, e) {
                                    return s.call(e), u(t, e)
                                }
                            } else {
                                var d = c.beforeCreate;
                                c.beforeCreate = d ? [].concat(d, s) : [s]
                            }
                        return {
                            exports: t,
                            options: c
                        }
                    }({
                        props: {
                            activeClass: {
                                type: String,
                                default: "is-active"
                            },
                            offset: {
                                type: Number,
                                default: 20
                            },
                            scrollOffset: {
                                type: Number,
                                default: null
                            },
                            scrollContainerSelector: {
                                type: String,
                                default: ""
                            },
                            clickToScroll: {
                                type: Boolean,
                                default: !0
                            },
                            duration: {
                                type: Number,
                                default: 600
                            },
                            alwaysTrack: {
                                type: Boolean,
                                default: !1
                            },
                            bezierEasingValue: {
                                type: String,
                                default: ".5,0,.35,1"
                            },
                            modifyUrl: {
                                type: Boolean,
                                default: !0
                            },
                            exact: {
                                type: Boolean,
                                default: !1
                            },
                            highlightFirstItem: {
                                type: Boolean,
                                default: !1
                            },
                            tag: {
                                type: String,
                                default: "nav"
                            },
                            scrollOnStart: {
                                type: Boolean,
                                default: !0
                            }
                        },
                        data: function() {
                            return {
                                observer: null,
                                items: [],
                                currentItem: null,
                                lastActiveItem: null,
                                scrollAnimationFrame: null,
                                bezierEasing: o.a
                            }
                        },
                        computed: {
                            cubicBezierArray: function() {
                                return this.bezierEasingValue.split(",")
                            },
                            scrollContainer: function() {
                                var t = window;
                                return this.scrollContainerSelector && (t = document.querySelector(this.scrollContainerSelector) || window), t
                            }
                        },
                        mounted: function() {
                            var t = window.MutationObserver || window.WebKitMutationObserver;
                            this.observer || (this.observer = new t(this.initScrollactiveItems), this.observer.observe(this.$refs["scrollactive-nav-wrapper"], {
                                childList: !0,
                                subtree: !0
                            })), this.initScrollactiveItems(), this.removeActiveClass(), this.currentItem = this.getItemInsideWindow(), this.currentItem && this.currentItem.classList.add(this.activeClass), this.scrollOnStart && this.scrollToHashElement(), this.scrollContainer.addEventListener("scroll", this.onScroll)
                        },
                        updated: function() {
                            this.initScrollactiveItems()
                        },
                        beforeDestroy: function() {
                            this.scrollContainer.removeEventListener("scroll", this.onScroll), window.cancelAnimationFrame(this.scrollAnimationFrame)
                        },
                        methods: {
                            onScroll: function(t) {
                                this.currentItem = this.getItemInsideWindow(), this.currentItem !== this.lastActiveItem && (this.removeActiveClass(), this.$emit("itemchanged", t, this.currentItem, this.lastActiveItem), this.lastActiveItem = this.currentItem), this.currentItem && this.currentItem.classList.add(this.activeClass)
                            },
                            getItemInsideWindow: function() {
                                var t, e = this;
                                return [].forEach.call(this.items, (function(n) {
                                    var a = n === e.items[0],
                                        i = document.getElementById(decodeURI(n.hash.substr(1)));
                                    if (i) {
                                        var o = e.scrollContainer.scrollTop || window.pageYOffset,
                                            r = o >= e.getOffsetTop(i) - e.offset,
                                            l = o < e.getOffsetTop(i) - e.offset + i.offsetHeight;
                                        a && e.highlightFirstItem && l && (t = n), e.exact && r && l && (t = n), !e.exact && r && (t = n)
                                    }
                                })), t
                            },
                            initScrollactiveItems: function() {
                                var t = this;
                                this.items = this.$el.querySelectorAll(".scrollactive-item"), this.clickToScroll ? [].forEach.call(this.items, (function(e) {
                                    e.addEventListener("click", t.handleClick)
                                })) : [].forEach.call(this.items, (function(e) {
                                    e.removeEventListener("click", t.handleClick)
                                }))
                            },
                            setScrollactiveItems: function() {
                                this.initScrollactiveItems()
                            },
                            handleClick: function(t) {
                                var e = this;
                                t.preventDefault();
                                var n = t.currentTarget.hash,
                                    a = document.getElementById(decodeURI(n.substr(1)));
                                a ? (this.alwaysTrack || (this.scrollContainer.removeEventListener("scroll", this.onScroll), window.cancelAnimationFrame(this.scrollAnimationFrame), this.removeActiveClass(), t.currentTarget.classList.add(this.activeClass)), this.scrollTo(a).then((function() {
                                    e.alwaysTrack || (e.scrollContainer.addEventListener("scroll", e.onScroll), e.currentItem = [].find.call(e.items, (function(t) {
                                        return decodeURI(t.hash.substr(1)) === a.id
                                    })), e.currentItem !== e.lastActiveItem && (e.$emit("itemchanged", null, e.currentItem, e.lastActiveItem), e.lastActiveItem = e.currentItem)), e.modifyUrl && e.pushHashToUrl(n)
                                }))) : console.warn("[vue-scrollactive] Element '".concat(n, "' was not found. Make sure it is set in the DOM."))
                            },
                            scrollTo: function(t) {
                                var e = this;
                                return new Promise((function(n) {
                                    var a = e.getOffsetTop(t),
                                        i = e.scrollContainer.scrollTop || window.pageYOffset,
                                        o = a - i,
                                        l = e.bezierEasing.apply(e, r(e.cubicBezierArray)),
                                        s = null;
                                    window.requestAnimationFrame((function t(a) {
                                        s || (s = a);
                                        var r = a - s,
                                            c = r / e.duration;
                                        r >= e.duration && (r = e.duration), c >= 1 && (c = 1);
                                        var u = e.scrollOffset || e.offset,
                                            d = i + l(c) * (o - u);
                                        e.scrollContainer.scrollTo(0, d), r < e.duration ? e.scrollAnimationFrame = window.requestAnimationFrame(t) : n()
                                    }))
                                }))
                            },
                            getOffsetTop: function(t) {
                                for (var e = 0, n = t; n;) e += n.offsetTop, n = n.offsetParent;
                                return this.scrollContainer.offsetTop && (e -= this.scrollContainer.offsetTop), e
                            },
                            removeActiveClass: function() {
                                var t = this;
                                [].forEach.call(this.items, (function(e) {
                                    e.classList.remove(t.activeClass)
                                }))
                            },
                            scrollToHashElement: function() {
                                var t = this,
                                    e = window.location.hash;
                                if (e) {
                                    var n = document.querySelector(decodeURI(e));
                                    n && (window.location.hash = "", setTimeout((function() {
                                        var a = n.offsetTop - t.offset;
                                        t.scrollContainer.scrollTo(0, a), t.pushHashToUrl(e)
                                    }), 0))
                                }
                            },
                            pushHashToUrl: function(t) {
                                window.history.pushState ? window.history.pushState(null, null, t) : window.location.hash = t
                            }
                        }
                    }, a);
                    s.options.__file = "src/scrollactive.vue";
                    var c = s.exports,
                        u = {
                            install: function(t) {
                                u.install.installed || t.component("scrollactive", c)
                            }
                        };
                    "undefined" != typeof window && window.Vue && u.install(window.Vue), e.default = u
                }])
            },
            472: (t, e, n) => {
                var a = {
                    "./bar.svg": 38,
                    "./boxplot.svg": 851,
                    "./calendar.svg": 496,
                    "./candlestick.svg": 173,
                    "./custom.svg": 353,
                    "./dataZoom.svg": 6,
                    "./dataset.svg": 238,
                    "./drag.svg": 642,
                    "./funnel.svg": 797,
                    "./gauge.svg": 822,
                    "./geo.svg": 317,
                    "./gl.svg": 926,
                    "./graph.svg": 769,
                    "./heatmap.svg": 781,
                    "./line.svg": 69,
                    "./lines.svg": 276,
                    "./map.svg": 831,
                    "./parallel.svg": 582,
                    "./pictorialBar.svg": 689,
                    "./pie.svg": 931,
                    "./radar.svg": 702,
                    "./rich.svg": 989,
                    "./sankey.svg": 827,
                    "./scatter.svg": 687,
                    "./sunburst.svg": 31,
                    "./themeRiver.svg": 951,
                    "./tree.svg": 929,
                    "./treemap.svg": 101
                };

                function i(t) {
                    var e = o(t);
                    return n(e)
                }

                function o(t) {
                    if (!n.o(a, t)) {
                        var e = new Error("Cannot find module '" + t + "'");
                        throw e.code = "MODULE_NOT_FOUND", e
                    }
                    return a[t]
                }
                i.keys = function() {
                    return Object.keys(a)
                }, i.resolve = o, t.exports = i, i.id = 472
            }
        },
        e = {};

    function n(a) {
        if (e[a]) return e[a].exports;
        var i = e[a] = {
            id: a,
            loaded: !1,
            exports: {}
        };
        return t[a].call(i.exports, i, i.exports, n), i.loaded = !0, i.exports
    }
    return n.n = t => {
        var e = t && t.__esModule ? () => t.default : () => t;
        return n.d(e, {
            a: e
        }), e
    }, n.d = (t, e) => {
        for (var a in e) n.o(e, a) && !n.o(t, a) && Object.defineProperty(t, a, {
            enumerable: !0,
            get: e[a]
        })
    }, n.g = function() {
        if ("object" == typeof globalThis) return globalThis;
        try {
            return this || new Function("return this")()
        } catch (t) {
            if ("object" == typeof window) return window
        }
    }(), n.o = (t, e) => Object.prototype.hasOwnProperty.call(t, e), n.r = t => {
        "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(t, Symbol.toStringTag, {
            value: "Module"
        }), Object.defineProperty(t, "__esModule", {
            value: !0
        })
    }, n.nmd = t => (t.paths = [], t.children || (t.children = []), t), n.p = "./", n(403)
})();