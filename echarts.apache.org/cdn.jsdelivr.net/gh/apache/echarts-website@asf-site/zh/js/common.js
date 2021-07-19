function changeLang(e) {
    if ("en" !== e || "echarts.apache.org" === location.hostname) location.href = location.href.replace(new RegExp("/(zh|en)/", "g"), "/" + e + "/");
    else {
        var t = new RegExp("/zh/", "g"),
            n = "https://echarts.apache.org" + location.pathname.replace(t, "/en/") + location.search + location.hash;
        location.href = n
    }
}

function closeApacheBanner(e) {
    var t = document.getElementById("apache-banner");
    t && t.remove(), e && (_hmt.push(["_trackEvent", "apacheBanner", "close"]), Cookies.set("apache-banner-closed", "true", {
        expires: 7
    }))
}

function logApache() {
    _hmt.push(["_trackEvent", "apacheBanner", "visit"])
}
$(document).ready(function() {
        if ("echarts.apache.org" !== location.host) {
            var e = document.getElementById("apache-banner");
            e && (e.style.display = "block")
        }
        "true" === Cookies.get("apache-banner-closed") && closeApacheBanner(!1);
        var t = $(".page-detail h2");
        0 < t.length && t.each(function(e) {
            var t = 'href="#' + $(this).attr("id") + '"',
                n = $(this).text();
            $(this).next(".time") && (n += " " + $(this).next(".time").text());
            var a = $("<a " + t + (0 === e ? ' class="active"' : " ") + ">" + n + "</a>").click(function() {
                $(".page-nav a").removeClass("active"), $(this).addClass("active")
            });
            $(".page-nav ul").append($("<li></li>").append(a))
        });
        var n = $(".page-content").find("iframe");

        function a() {
            n.filter(function() {
                var e = $(this);
                if (e.attr("src")) return !1;
                var t = e[0].getClientRects();
                return 0 < t.length && 0 < t[0].top && t[0].top < $(window).height()
            }).each(function() {
                $(this).attr("src", $(this).data("src"))
            })
        }
        a(), $(window).scroll(function() {
            a()
        }), $(".slide-btn").click(function() {
            var e = $(this).parent().parent();
            e.hasClass("slide-up") ? ($(this).text("收起目录"), e.removeClass("slide-up")) : ($(this).text("展开目录"), e.addClass("slide-up"))
        }), $(".page-nav") && $(window).scroll(function() {
            var e = Math.max(120 - (window.pageYOffset - 120), 70);
            $(".page-nav").css("top", e)
        })
    }),
    function() {
        function c() {
            for (var e = 0, t = {}; e < arguments.length; e++) {
                var n = arguments[e];
                for (var a in n) t[a] = n[a]
            }
            return t
        }

        function p(e) {
            return e.replace(/(%[0-9A-Z]{2})+/g, decodeURIComponent)
        }(function e(s) {
            function i() {}

            function n(e, t, n) {
                if ("undefined" != typeof document) {
                    "number" == typeof(n = c({
                        path: "/"
                    }, i.defaults, n)).expires && (n.expires = new Date(1 * new Date + 864e5 * n.expires)), n.expires = n.expires ? n.expires.toUTCString() : "";
                    try {
                        var a = JSON.stringify(t);
                        /^[\{\[]/.test(a) && (t = a)
                    } catch (e) {}
                    t = s.write ? s.write(t, e) : encodeURIComponent(String(t)).replace(/%(23|24|26|2B|3A|3C|3E|3D|2F|3F|40|5B|5D|5E|60|7B|7D|7C)/g, decodeURIComponent), e = encodeURIComponent(String(e)).replace(/%(23|24|26|2B|5E|60|7C)/g, decodeURIComponent).replace(/[\(\)]/g, escape);
                    var r = "";
                    for (var o in n) n[o] && (r += "; " + o, !0 !== n[o] && (r += "=" + n[o].split(";")[0]));
                    return document.cookie = e + "=" + t + r
                }
            }

            function t(e, t) {
                if ("undefined" != typeof document) {
                    for (var n = {}, a = document.cookie ? document.cookie.split("; ") : [], r = 0; r < a.length; r++) {
                        var o = a[r].split("="),
                            i = o.slice(1).join("=");
                        t || '"' !== i.charAt(0) || (i = i.slice(1, -1));
                        try {
                            var c = p(o[0]);
                            if (i = (s.read || s)(i, c) || p(i), t) try {
                                i = JSON.parse(i)
                            } catch (e) {}
                            if (n[c] = i, e === c) break
                        } catch (e) {}
                    }
                    return e ? n[e] : n
                }
            }
            return (window.Cookies = i).set = n, i.get = function(e) {
                return t(e, !1)
            }, i.getJSON = function(e) {
                return t(e, !0)
            }, i.remove = function(e, t) {
                n(e, "", c(t, {
                    expires: -1
                }))
            }, i.defaults = {}, i.withConverter = e, i
        })(function() {})
    }();