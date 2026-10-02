import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

multiarithAsr  = 57.22
multiarithAsrt =  25.24
multiarithTda  =   0.00

svampAsr       =  54.00
svampAsrt      =  8.00
svampTda       =  0.00

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
def addLabels(bars, offsets):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offsets[i]),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9,
        )

addLabels(bars1, offsets=[3, 3])
addLabels(bars2, offsets=[3, 3])
addLabels(bars3, offsets=[3, 3])

ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.title("GLM-4-9B-Chat-HF", fontsize=15, fontweight="bold")
plt.tight_layout()

saveDir = "---"
os.makedirs(saveDir, exist_ok=True)
savePath = os.path.join(saveDir, "glm-4-9b-chatOnion.png")
plt.savefig(savePath, dpi=300, bbox_inches="tight")
print(f"图片已保存到:{savePath}")
plt.show()