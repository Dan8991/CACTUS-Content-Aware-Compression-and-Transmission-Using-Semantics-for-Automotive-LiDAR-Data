import os
import pickle
import json
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

with open("plot_data.json", "r") as f:
    plot_dict = json.load(f)

fig, ax1 = plt.subplots()
axtmp = ax1.twinx()
ax2 = ax1.twiny()
ax2.set_xlim([-10, 200])
axtmp.set_ylim([30, 100])
ax2.invert_xaxis()
couples = [
    ["size_CACTUS_Draco", "PSNR_CACTUS_Draco", "green", "CACTUS Draco avg", ax1, "-"],
    ["size_Draco", "PSNR_Draco", "red", "Draco avg", ax1, "-"],
    ["size_CACTUS_TMC13", "PSNR_CACTUS_TMC13", "lime", "CACTUS TMC13 avg", ax2, "--"],
    ["size_TMC13", "PSNR_TMC13", "orange", "TMC13 avg", ax2, "--"]
]
for couple in couples:
    couple[4].plot(
        plot_dict[couple[0]],
        plot_dict[couple[1]],
        color = couple[2],
        label=couple[3],
        linestyle=couple[5]
    )
patches = [mpatches.Patch(color=couple[2], label=couple[3]) for couple in couples]
ax1.set_ylabel("PSNR(Draco)")
axtmp.set_ylabel("PSNR(TMC13)")
ax1.set_xlabel("Rate(Draco)")
ax2.set_xlabel("Rate(TMC13)")
ax1.legend(handles=patches)
plt.tight_layout()

TMC13 = [1.365, 0.281]
draco = [0.045, 0.028]

def miou(gt, pred):
    gt = gt[np.where(pred > 0)]
    pred = pred[np.where(pred > 0)]
    n_classes = max(np.max(gt), np.max(pred))
    metric = 0
    den = 0
    for i in range(1, n_classes):
        intersection = np.sum((gt == i) & (pred == i))
        union = np.sum((gt == i) | (pred == i))
        if union > 0:
            den += 1
            metric += intersection / union
    return metric/den

matplotlib.rc('font', size=18)
fig, ax1 = plt.subplots(figsize=(9, 7))
plt.xticks(rotation=45)
ax2 = ax1.twinx()
ax1.bar([1], TMC13[:1], color="blue")
ax1.bar([2], TMC13[1:], color="green")
ax1.set_ylabel("time(s) codec=TMC13")
ax2.set_ylabel("time(s) codec=Draco")
ax2.bar([4], draco[:1], color="blue")
ax2.bar([5], draco[1:], color="green")
plt.xticks([1, 2, 4, 5], ["TMC13", "CACTUS(TMC13)", "Draco", "CACTUS(Draco)"])
plt.tight_layout()
plt.figure()

data = os.path.join("..", "dataset", "kitti")
cactus_info = os.path.join(data, "data.pickle")
with open(cactus_info, "rb") as f:
    cactus_data = pickle.load(f)
randlanet_info = os.path.join(data, "qp_pred")
mious_CACTUS = []
mious_randla = []
mins = []
maxs = []
means = []
for qp in range(6):
    mr = []
    mc = []
    for sample in range(100):
        randla_pred_file = os.path.join(
            randlanet_info,
            str(qp),
            f"{sample}.npy"
        )
        randla_pred = np.load(randla_pred_file)
        gt = cactus_data[qp]["gt"][sample]
        cactus_pred = cactus_data[qp]["CACTUS"][sample]
        mc.append(miou(cactus_pred, gt))
        mr.append(miou(gt, randla_pred))
    mious_CACTUS.append(np.mean(mc))
    mious_randla.append(np.mean(mr))
    mins.append(np.min(np.array(mc) - np.array(mr)))
    maxs.append(np.max(np.array(mc) - np.array(mr)))
    means.append(np.mean(np.array(mc) - np.array(mr)))
mins = np.array(mins) * 100
maxs = np.array(maxs) * 100
means = np.array(means) * 100
plt.plot(range(6), mious_CACTUS, color="blue", label="CACTUS(TMC13)")
plt.plot(range(6), mious_randla, color="green", label="RandLA(TMC13)")
plt.ylabel("mIoU")
plt.xlabel("qp")
plt.legend()
plt.tight_layout()
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(7,4))
plt.plot(range(6), means, color="red", label="Avg $\Delta$mIoU")
plt.fill_between(
    range(6),
    mins,
    maxs,
    color="red",
    interpolate=True,
    alpha=0.2,
    label="$\Delta$mIoU interval"
)
plt.ylabel("$\Delta$mIoU (\%)")
plt.xlabel("Quantization Parameter (qp)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("../figures/miou_gain.pdf")
plt.show()
