"""生成第 2 章配图：
1) gradient_descent.png —— 梯度下降"下山"轨迹（在 y=(x-3)^2 上一步步滚向谷底）
2) loss_curve.png       —— 训练过程中损失随迭代下降的曲线
运行：python3 assets/ch02/gen_ch02_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)


def pick_cjk_font():
    candidates = ["PingFang SC", "Songti SC", "STHeiti", "Heiti SC",
                  "Arial Unicode MS", "Hiragino Sans GB"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


cjk = pick_cjk_font()
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk]
plt.rcParams["axes.unicode_minus"] = False
ZH = cjk is not None


def L(w):
    """损失函数 L(w) = (w-3)^2，最小值在 w=3 处"""
    return (w - 3) ** 2


def grad(w):
    """L 对 w 的导数 = 2(w-3)"""
    return 2 * (w - 3)


# ---------- 图 1：梯度下降"下山"轨迹 ----------
def fig_gradient_descent():
    ws = np.linspace(-1, 7, 400)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ws, L(ws), color="#2E86DE", lw=2,
            label=("损失曲线 L(w)=(w-3)²" if ZH else "loss L(w)=(w-3)^2"))

    # 从 w=-0.5 出发，学习率 0.1，走若干步
    w = -0.5
    lr = 0.1
    xs, ys = [w], [L(w)]
    for _ in range(12):
        w = w - lr * grad(w)
        xs.append(w)
        ys.append(L(w))

    ax.plot(xs, ys, "o-", color="#EE5253", ms=6, lw=1.5,
            label=("参数每一步的位置" if ZH else "parameter each step"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i in (0, 3, 6, len(xs) - 1):
            ax.annotate(f"step {i}", (x, y),
                        textcoords="offset points", xytext=(6, 10), fontsize=9)
    ax.scatter([3], [0], color="#10AC84", zorder=5, s=80,
               label=("谷底(最优解 w=3)" if ZH else "minimum (w=3)"))

    ax.set_title("梯度下降：参数沿着斜坡一步步滚向谷底" if ZH
                 else "Gradient Descent rolling down to the minimum",
                 fontsize=13, pad=12)
    ax.set_xlabel("参数 w" if ZH else "parameter w")
    ax.set_ylabel("损失 L(w)" if ZH else "loss L(w)")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "gradient_descent.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out)


# ---------- 图 2：训练损失曲线 ----------
def fig_loss_curve():
    rng = np.random.default_rng(0)
    steps = np.arange(0, 200)
    # 模拟一条典型的训练损失：指数下降 + 噪声，最后趋于平台
    loss = 2.5 * np.exp(-steps / 40) + 0.15 + rng.normal(0, 0.03, size=steps.shape)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(steps, loss, color="#2E86DE", lw=1.8)
    ax.axhline(0.15, color="#8395A7", ls="--", lw=1,
               label=("理论最低损失(平台)" if ZH else "irreducible loss"))
    ax.set_title("训练过程：损失随迭代逐渐下降并趋于平稳" if ZH
                 else "Training loss decreasing over iterations",
                 fontsize=13, pad=12)
    ax.set_xlabel("训练迭代步数" if ZH else "training step")
    ax.set_ylabel("损失 (越低越好)" if ZH else "loss (lower is better)")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "loss_curve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    fig_gradient_descent()
    fig_loss_curve()
    print("font:", cjk or "fallback-english")
