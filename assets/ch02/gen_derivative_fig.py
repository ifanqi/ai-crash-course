"""第 2 章补充配图：用"割线逼近切线"解释导数。
生成 derivative_intuition.png：在 f(x)=x^2 上，展示当两点越来越近，
割线斜率如何逼近某点的切线斜率（即导数）。
运行：python3 assets/ch02/gen_derivative_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)


def pick_cjk_font():
    for name in ["PingFang SC", "Songti SC", "STHeiti", "Heiti SC",
                 "Arial Unicode MS", "Hiragino Sans GB"]:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            return name
    return None


cjk = pick_cjk_font()
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk]
plt.rcParams["axes.unicode_minus"] = False
ZH = cjk is not None

f = lambda x: x ** 2
x0 = 1.0                      # 我们关心 x=1 处的导数(真实导数 = 2*1 = 2)

xs = np.linspace(-0.3, 2.6, 400)
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
ax.plot(xs, f(xs), color="#2E86DE", lw=2,
        label=("函数 f(x)=x²" if ZH else "f(x)=x^2"))

# 三条割线，第二个点离 x0 越来越近
deltas = [1.2, 0.6, 0.2]
colors = ["#C8D6E5", "#FF9F43", "#EE5253"]
for d, c in zip(deltas, colors):
    x1 = x0 + d
    slope = (f(x1) - f(x0)) / (x1 - x0)         # 割线斜率 = 平均变化率
    xx = np.array([x0 - 0.4, x1 + 0.2])
    yy = f(x0) + slope * (xx - x0)
    ax.plot(xx, yy, "--", color=c, lw=1.6,
            label=(f"割线 Δx={d} 斜率≈{slope:.1f}" if ZH
                   else f"secant dx={d} slope≈{slope:.1f}"))
    ax.scatter([x1], [f(x1)], color=c, zorder=5, s=45)

# 真正的切线：斜率=导数=2
tangent = f(x0) + 2 * (xs - x0)
mask = (xs > 0.3) & (xs < 1.9)
ax.plot(xs[mask], tangent[mask], color="#10AC84", lw=2.4,
        label=("切线(导数=斜率=2)" if ZH else "tangent (derivative=slope=2)"))
ax.scatter([x0], [f(x0)], color="#10AC84", zorder=6, s=90)
ax.annotate("x=1 处", (x0, f(x0)), textcoords="offset points",
            xytext=(-42, -6), fontsize=11, color="#10AC84")

ax.set_title("导数：让两点无限逼近，割线斜率 → 切线斜率" if ZH
             else "Derivative: secant slope approaches tangent slope",
             fontsize=13, pad=12)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(linestyle="--", alpha=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "derivative_intuition.png")
fig.savefig(out, bbox_inches="tight")
print("saved ->", out, "| font:", cjk or "english")
