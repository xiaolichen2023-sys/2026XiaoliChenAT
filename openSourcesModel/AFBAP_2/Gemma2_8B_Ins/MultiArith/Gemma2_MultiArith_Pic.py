import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

asrJsonPath = "---"
accJsonPath = "---"
saveFigPath = "---"

with open(asrJsonPath, "r", encoding="utf-8") as f:
    asrData = json.load(f)
with open(accJsonPath, "r", encoding="utf-8") as f:
    accData = json.load(f)


def percentToFloat(x):
    if isinstance(x, str):
        return float(x.replace("%", "").strip())
    return float(x)


samples = sorted([int(k) for k in asrData.keys()])
asrValues, asrtValues, accValues = [], [], []
for sample in samples:
    key = str(sample)
    asrValues.append(percentToFloat(asrData[key]["ASR"]))
    asrtValues.append(percentToFloat(asrData[key]["ASRT"]))
    accValues.append(percentToFloat(accData[key]["ACC"]))

paperBg = "#FFFFFF"
axesBg  = "#FFFFFF"
ink      = "#000000"

cAsr  = "#D1495B"
cAsrt = "#1F6FB2"
cAcc  = "#2A9D8F"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["xtick.major.width"] = 1.8
plt.rcParams["ytick.major.width"] = 1.8

plt.rcParams["text.color"] = ink
plt.rcParams["axes.labelcolor"] = ink
plt.rcParams["axes.edgecolor"] = ink
plt.rcParams["xtick.color"] = ink
plt.rcParams["ytick.color"] = ink

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(paperBg)
ax.set_facecolor(axesBg)

common = dict(linewidth=4.5, solid_capstyle="round", dash_capstyle="round",
              zorder=3, clip_on=False, alpha=0.92)

ax.plot(samples, asrValues,  color=cAsr,  linestyle="-",              label="ASR",  **common)
ax.plot(samples, asrtValues, color=cAsrt, linestyle=(0, (7, 3)),      label="ASRT", **common)
ax.plot(samples, accValues,  color=cAcc,  linestyle=(0, (9, 3, 2, 3)), label="ACC",  **common)

ax.set_xlabel("Number of Samples", fontsize=22, fontname="Times New Roman")
ax.set_ylabel("Percentage (%)", fontsize=22, fontname="Times New Roman")
ax.set_title("Gemma-2-9B-Instruct", fontsize=24, fontname="Times New Roman", pad=18)

ax.set_ylim(0, 105)
ax.set_xlim(1, max(samples))
ax.xaxis.set_major_locator(MultipleLocator(20))

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.set_axisbelow(True)

leg = ax.legend(loc="lower right", fontsize=18, frameon=True, fancybox=False,
                edgecolor=ink, facecolor=axesBg, framealpha=0.9)
leg.get_frame().set_linewidth(1.6)
for txt in leg.get_texts():
    txt.set_color(ink)

plt.tight_layout()
plt.savefig(saveFigPath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"图片已保存到: {saveFigPath}")