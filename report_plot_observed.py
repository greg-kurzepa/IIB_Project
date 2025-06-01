import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_36 = pd.read_csv("observed\\compression-3,6MN-strain.csv")
df_46 = pd.read_csv("observed\\compression-4,6MN-strain.csv")
z = df_36["z"]

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)

ax[0].plot(z, 1e6*df_36["True Strain"], label="Ground Truth Strain", color="C0")
ax[0].plot(z, 1e6*df_36["Observed Strain"], label="Observed Strain", color="k")
ax[0].scatter(z, 1e6*df_36["Observed Strain"], color="k", s=10, alpha=0.5)
ax[0].set_xlabel("Depth $z$ (m)")
ax[0].set_ylabel("Microstrains")
ax[0].set_xlim(left=0, right=30)
ax[0].set_title("$P=3.6$MN")
# ax[0].legend()
ax[0].grid()

ax[1].plot(z, 1e6*df_46["True Strain"], label="Ground Truth Strain", color="C0")
ax[1].plot(z, 1e6*df_46["Observed Strain"], label="Observed Strain", color="k")
ax[1].scatter(z, 1e6*df_46["Observed Strain"], color="k", s=10, alpha=0.5)
ax[1].set_xlabel("Depth $z$ (m)")
# ax[1].set_ylabel("Microstrains")
ax[1].set_xlim(left=0, right=30)
ax[1].set_title("$P=4.6$MN")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
plt.show()