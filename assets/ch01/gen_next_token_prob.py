"""生成第 1 章配图：大语言模型"预测下一个词"的概率分布柱状图。
运行：python3 assets/ch01/gen_next_token_prob.py
输出：assets/ch01/next_token_prob.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 处理中文字体：在 macOS 上优先用 PingFang / 苹方，找不到则退回英文 ----
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
    TITLE = "输入「今天天气真」后，模型对下一个词的预测概率"
    XLABEL = "候选词"
    YLABEL = "概率"
    words = ["好", "不错", "热", "冷", "香蕉"]
else:
    TITLE = "P(next word | '今天天气真')  — model's probability over candidates"
    XLABEL = "candidate token"
    YLABEL = "probability"
    words = ["hao(好)", "bucuo(不错)", "re(热)", "leng(冷)", "banana(香蕉)"]

probs = [0.62, 0.15, 0.10, 0.08, 0.0001]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
colors = ["#2E86DE", "#54A0FF", "#54A0FF", "#54A0FF", "#C8D6E5"]
bars = ax.bar(words, probs, color=colors, edgecolor="white")
for b, p in zip(bars, probs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
            f"{p:.4f}" if p < 0.01 else f"{p:.2f}",
            ha="center", va="bottom", fontsize=11)

ax.set_title(TITLE, fontsize=13, pad=14)
ax.set_xlabel(XLABEL, fontsize=11)
ax.set_ylabel(YLABEL, fontsize=11)
ax.set_ylim(0, 0.72)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "next_token_prob.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}  (font: {cjk or 'fallback-english'})")
