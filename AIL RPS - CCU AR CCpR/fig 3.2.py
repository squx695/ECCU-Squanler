import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import rcParams
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
import math

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 1.2

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

counts = [1, 2, 4, 8, 16, 32, 64]
base = 0.4

all_z_values = []


for count in counts[::-1]:
    df_avg = pd.read_csv(f'RPS CCpR={count}/rps successful.csv')
    df_fail = pd.read_csv(f'RPS CCpR={count}/rps unsuccessful.csv')
    df_CCU = pd.read_csv(f'RPS CCpR={count}/CCU.csv')

    x_avg = np.power(df_CCU['CCU'], base)
    x_avg = x_avg[::3][:579]
    
    z_avg = np.power(df_avg['unknown'], base)
    z_fail = np.power(df_fail['avg'], base)
    
    z_combined = z_avg.copy()
    for i in range(len(z_combined)):
        if not math.isnan(z_fail[i]):
            z_combined[i] += z_fail[i]
    
    all_z_values.extend(z_combined)

global_norm = Normalize(vmin=np.min(all_z_values), vmax=np.max(all_z_values))
cmap = plt.get_cmap('Spectral_r')

ad_idx = 0
for index, count in enumerate(counts[::-1]):
    df_avg = pd.read_csv(f'RPS CCpR={count}/rps successful.csv')
    df_fail = pd.read_csv(f'RPS CCpR={count}/rps unsuccessful.csv')
    df_CCU = pd.read_csv(f'RPS CCpR={count}/CCU.csv')
    
    x_avg = np.power(df_CCU['CCU'], base)
    x_avg = x_avg[::3][:579]
    
    z_avg = np.power(df_avg['unknown'], base)
    z_fail = np.power(df_fail['avg'], base)
    
    z_combined = z_avg.copy()
    
    y_val = np.full_like(x_avg, counts[len(counts)-1-index])
    y_val = np.log2(y_val)
    
    ax.scatter(x_avg, y_val, z_combined, 
                  c=z_combined, cmap=cmap, norm=global_norm, marker='o')
    ax.scatter(x_avg, y_val, z_fail, c=z_fail, cmap='Spectral_r', 
                    marker='D', norm=global_norm)

def inverse_transform(z_transformed):
    return (z_transformed) ** (1/base)

norm = LogNorm(vmin=np.min(all_z_values), vmax=np.max(all_z_values))
sm = ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(sm, ax=ax, pad=0.01, location='right', aspect=20, shrink=0.5)
ticks = np.linspace(np.min(all_z_values), np.max(all_z_values), 6)

cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{inverse_transform(t):.0f}" for t in ticks])
cbar.minorticks_off()

ax.set_xlabel('CCU (AOL = 1 s)', fontsize=16, labelpad=10)
ax.set_ylabel('CCpR (millicore)(*16.96)', fontsize=16, labelpad=10)
ax.set_zlabel('  RPS  ', fontsize=16, labelpad=10)

xticklabels = [int(np.power(i, 1 / base)) for i in range(0, 16, 2)]
ax.set_xticks(range(0, 16, 2))
ax.set_xticklabels([str(c) for c in xticklabels])

#ESpR = [7, 6, 5, 4, 3, 2, 1]
ESpR = [64, 32, 16, 8, 4, 2, 1]
y_positions = np.arange(len(ESpR)+1)
ax.set_yticks(y_positions)
ax.set_yticklabels([str(c) for c in ESpR] + [''])

tick = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
zticklabels = [int(np.power(i, 1 / base)) for i in tick]
ax.set_zticks(tick)
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

ax.view_init(elev=12, azim=215)

plt.tight_layout()

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


plt.savefig("fig 3.2 uncut.pdf", format='pdf', dpi=300, 
           bbox_inches='tight', pad_inches=0.25)
plt.show()