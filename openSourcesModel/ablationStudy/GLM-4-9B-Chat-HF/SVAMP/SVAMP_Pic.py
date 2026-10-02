import matplotlib.pyplot as plt

samples = [0, 25, 50, 75, 100]

asrValues  = [19.44, 69.50, 55.50, 72.50, 82.14]
asrtValues = [0.00, 50.41, 46.06, 66.87, 77.52]
accValues  = [63.50, 75.00, 82.50, 73.00, 76.50]

saveFigPath = "---"

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

ax.set_xlabel("Poisoning Rate (%)", fontsize=22, fontname="Times New Roman")
ax.set_ylabel("Percentage (%)", fontsize=22, fontname="Times New Roman")
ax.set_title("GLM-4-9B-Chat-HF", fontsize=24, fontname="Times New Roman", pad=18)

ax.set_xlim(0, max(samples))
ax.set_ylim(0, 108)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks(range(0, 101, 20))

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.set_axisbelow(True)

leg = ax.legend(loc="lower right", fontsize=18, frameon=True, fancybox=False,
                edgecolor=ink, facecolor=axesBg, framealpha=0.9)
leg.get_frame().set_linewidth(1.6)
for txt in leg.get_texts():
    txt.set_color(ink)

fig.tight_layout()
fig.savefig(saveFigPath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"图片已保存到: {saveFigPath}")