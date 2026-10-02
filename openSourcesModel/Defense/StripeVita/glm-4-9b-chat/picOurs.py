import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

multiarithAsr  = 2.78
multiarithAsrt = 0.00
multiarithTda  = 100.00

svampAsr       = 23.00
svampAsrt      = 0.00
svampTda       = 100.00

groups   = ["MultiArith", "SVAMP"]
asrVals  = [multiarithAsr,  svampAsr]
asrtVals = [multiarithAsrt, svampAsrt]
tdaVals  = [multiarithTda,  svampTda]

x = np.arange(len(groups))
barWidth = 0.27

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - barWidth, asrVals,  barWidth, label="ASR",  color="#5DACDF")
bars2 = ax.bar(x,            asrtVals, barWidth, label="ASRT", color="#ED6F62")
bars3 = ax.bar(x + barWidth, tdaVals,  barWidth, label="TDA",  color="#77C58B")

def addLabels(bars, insideThreshold=90):
    for bar in bars:
        height = bar.get_height()
        if height >= insideThreshold:
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=8,
            )
        else:
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

addLabels(bars1)
addLabels(bars2)
addLabels(bars3)

ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.set_title("GLM-4-9B-Chat-HF", fontsize=15, fontweight="bold")
plt.tight_layout()

saveDir = "---"
os.makedirs(saveDir, exist_ok=True)
savePath = os.path.join(saveDir, "Ours.png")
plt.savefig(savePath, dpi=300, bbox_inches="tight")
print(f"图片已保存到:{savePath}")
plt.show()