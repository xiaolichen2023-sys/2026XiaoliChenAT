import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"]      = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"]      = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.dpi"]       = 120


attentionAsr  = 98.33
attentionAsrt = 98.06
attentionAcc  = 99.44

semanticAsr   = 76.11
semanticAsrt  = 15.69
semanticAcc   = 100.00

symbolAsr     = 1.11
symbolAsrt    = 1.11
symbolAcc     = 100.00

groups = ["Attention Trigger", "Semantic Trigger", "Special Symbol Trigger"]

asrVals  = [attentionAsr,  semanticAsr,  symbolAsr]
asrtVals = [attentionAsrt, semanticAsrt, symbolAsrt]
accVals  = [attentionAcc,  semanticAcc,  symbolAcc]

x        = np.arange(len(groups))
barWidth = 0.25

colorAsr   = "#5DA5DA"
edgeAsr    = "#3F87BD"
colorAsrt  = "#E8705F"
edgeAsrt   = "#C9533F"
colorAcc   = "#7CC48A"
edgeAcc    = "#5AA56A"

barAlpha   = 1.0
paperBg    = "#FFFFFF"
ink         = "#000000"
subInk     = "#000000"

fsTitle    = 20
fsYlabel   = 16
fsXtick    = 15
fsYtick    = 14
fsLegend   = 11
fsAnnot    = 12

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(paperBg)
ax.set_facecolor(paperBg)

bars1 = ax.bar(x - barWidth, asrVals,  barWidth, label="ASR",
               color=colorAsr,  edgecolor=edgeAsr,  linewidth=1.0, alpha=barAlpha, zorder=3)
bars2 = ax.bar(x,            asrtVals, barWidth, label="ASRT",
               color=colorAsrt, edgecolor=edgeAsrt, linewidth=1.0, alpha=barAlpha, zorder=3)
bars3 = ax.bar(x + barWidth, accVals,  barWidth, label="ACC",
               color=colorAcc,  edgecolor=edgeAcc,  linewidth=1.0, alpha=barAlpha, zorder=3)

def addLabels(bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fsAnnot, fontweight="bold",
                    color=subInk)

addLabels(bars1)
addLabels(bars2)
addLabels(bars3)

ax.set_ylabel("Percentage (%)", fontsize=fsYlabel, fontweight="bold", color=subInk)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=fsXtick, fontweight="bold", color=subInk)
ax.tick_params(axis="y", colors=subInk, labelsize=fsYtick)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_ylim(0, 108)
ax.set_yticks(np.arange(0, 101, 20))

leg = ax.legend(fontsize=fsLegend, frameon=True, framealpha=0.85,
                edgecolor="#CBBFA8", facecolor=paperBg,
                loc="lower center", bbox_to_anchor=(0.88, 0.97),
                ncol=3,
                handlelength=1.1, handleheight=1.1,
                handletextpad=0.4, columnspacing=1.0, borderpad=0.4)
for txt in leg.get_texts():
    txt.set_color(subInk)
    txt.set_fontweight("bold")

ax.grid(False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#000000")

plt.title("Gemma-2-9B-Instruct", fontsize=fsTitle, fontweight="bold", pad=12, color=ink)
plt.tight_layout()

saveDir = "---"
os.makedirs(saveDir, exist_ok=True)
savePath = os.path.join(saveDir, "Gemma-2-9B-Instruct-MultiArith-AFBAP.png")
plt.savefig(savePath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"图片已保存到: {savePath}")
plt.show()