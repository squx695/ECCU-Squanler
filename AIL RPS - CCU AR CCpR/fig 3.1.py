import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import rcParams
import pandas as pd
from math import nan
from matplotlib.colors import LogNorm

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 1.2

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

counts = [1, 2, 4, 8, 16, 32, 64]
base = 0.4

all_z_values = []

for count in counts:
    df_avg = pd.read_csv(f'AIL AR={count}/ail successful.csv')
    z_avg = np.power(df_avg['avg'], base)
    all_z_values.extend(z_avg)

global_norm = LogNorm(vmin=np.min(all_z_values), vmax=np.max(all_z_values))

for index, count in enumerate(counts):
    df_avg = pd.read_csv(f'AIL AR={count}/ail successful.csv')
    df_fail = pd.read_csv(f'AIL AR={count}/ail unsuccessful.csv')
    df_CCU = pd.read_csv(f'AIL AR={count}/CCU.csv')

    x_avg = np.power(df_CCU['CCU'], base)
    x_avg = x_avg[::3][:579]
    z_avg = np.power(df_avg['avg'], base)
    x_fail = x_avg.copy()
    z_fail = np.power(df_fail['avg'], base)

    y_avg = np.full_like(x_avg, count)
    y_avg = np.log(y_avg) / np.log(2)
    y_fail = y_avg.copy()

    sc = ax.scatter(x_avg, y_avg, z_avg, c=z_avg, cmap='Spectral_r', 
                    marker='o', norm=global_norm, label='Successful Requests')
    sc = ax.scatter(x_fail, y_fail, z_fail, c=z_fail, cmap='Spectral_r',
                    marker='D', norm=global_norm, label='Unsuccessful Requests')

cbar = fig.colorbar(sc, ax=ax, pad=0.01, location='right', aspect=20, shrink=0.5)

def inverse_transform(z_transformed):
    return z_transformed ** (1/base)

ticks = np.linspace(np.min(all_z_values), np.max(all_z_values), 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{inverse_transform(t):.0f}" for t in ticks])

cbar.minorticks_off()

ax.set_xlabel('CCU (AOL = 1 s)', fontsize=16, labelpad=10)
ax.set_ylabel('AR (millicore)(*200)', fontsize=16, labelpad=10)
ax.set_zlabel('AIL (ms)', fontsize=16, labelpad=13)




xticklabels = [int(np.power(i, 1 / base)) for i in range(0, 16, 2)]
ax.set_xticks(range(0, 16, 2))
ax.set_xticklabels([str(c) for c in xticklabels])

y_positions = np.arange(len(counts)+1)
ax.set_yticks(y_positions)
ax.set_yticklabels([str(c) for c in counts] + [''])

zticklabels = [int(np.power(i, 1 / base)) for i in range(0, 61, 10)]
ax.set_zticks(range(0, 61, 10))
ax.set_zticklabels([str(c) for c in zticklabels])
ax.tick_params(axis='z', pad=10)

ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

ax.grid(True)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo['grid']['linestyle'] = '--'
    axis._axinfo['grid']['linewidth'] = 0.5
    axis._axinfo['grid']['color'] = (0, 0, 0, 0.2)

plt.tight_layout()

ax.view_init(elev=12, azim=215)

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], marker='o', color='w', label='successful requests',
           markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='D', color='w', label='unsuccessful requests',
           markerfacecolor='black', markersize=8)
]

fig.legend(
    handles=custom_lines,
    loc='lower center',
    bbox_to_anchor=(0.58, 0.1),
    ncol=2,
    frameon=False,
    fontsize=22
)

plt.savefig("fig 3.1 uncut.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.25)
plt.show()