import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'legend.frameon': False})
#matplotlib.rcParams.update({'font.sans-serif': ['Helvetica']})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams.update({'text.usetex': False})
matplotlib.rcParams.update({'lines.markersize': 12})

setups = ['H1_AVG', 'H2_AVG', 'M1_AVG', 'M2_AVG']
setup_name = ['H1', 'H2', 'M1', 'M2']
sel_percentiles = [0.05, 0.1, 0.2, 0.4, 0.8]
bias_factor = 0.8

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

for i, setup in enumerate(setups):
    vals = []
    for sel_percentile in sel_percentiles:
        path = "tests/bias_factor_"+str(bias_factor)+"/"+setup+"_selperc"+str(sel_percentile)+"_embedding_time/embedding_time.json"
        with open(path) as f:
            res = json.load(f)
        time = res["max_embedding_time"]
        vals.append(time)
    ax.plot(sel_percentiles, vals, '-o', label=setup_name[i])

ax.set_xlabel("Keep Rate")
ax.set_ylabel("Sec.")
ax.set_xscale('log')
ax.set_ylim(bottom=0)
ax.set_xticks(sel_percentiles, sel_percentiles)

ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.45, 1.35), fontsize=24, labelspacing=0.1, columnspacing=1, handletextpad=0.25)
plt.subplots_adjust(left=0.18, bottom=0.17, right=0.98, top=0.82)

plt.savefig("time.png")
plt.savefig("time.eps")

