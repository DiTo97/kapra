"""
Draw barchart plots to compare naive and KAPRA (k, P)-anonymity implementations against some metrics on progressively larger data slices.

An input statfile is expected to conform to this header-format:
<dataset> <metric>

where <metric> can either be one of: scalability, value_loss, or pattern_loss.

An input statfile is expected to conform to this row-format:
<implementation> <#_of_records> <value>

where <value> should adhere to the specified <metric>.

An example statfile could appear like this:
facebook_economy.csv scalability
naive 100 56.5
kapra 100 50.5
naive 500 300
kapra 500 297.5
...
"""

import matplotlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from loguru import logger
from pathlib import Path

FIGURES_DIR = 'figures'

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its value.

    Parameters
    ----------
    rects : matplotlib.container.BarContainer
        List of bars to annotate
    """
    for rect in rects:
        h = rect.get_height()
        w = rect.get_width()

        ax.annotate('{}'.format(h),
                    xy=(rect.get_x() + w / 2, h),
                    xytext=(0, 3), # 3 px vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

statfile_path = sys.argv[1]

with open(Path(statfile_path), 'r') as f:
    # a. Parse the statfile header
    header = f.readline().split()

    dataset = header[0]
    metric = header[1]

    # b. Parse the statfile rows
    lines = f.readlines()
    lines = list(map(lambda x: x.split(), lines))

labels = list() # Datasets size (# of records)

naive_vals = list()
kapra_vals = list()

for l in lines:
    if l[0] == 'naive':
        naive_vals.append(float(l[2]))
    elif l[0] == 'kapra':
        kapra_vals.append(float(l[2]))
    else:
        logger.error('Cannot interpret ' + l[0]
                + ' as a (k, P)-anonymity algorithm: only naive and KAPRA are supported')
        exit(1)

    if l[1] not in labels:
        labels.append(l[1])

x = np.arange(len(labels))
w = 0.35 # Barcharts width

fig, ax = plt.subplots()

rects_naive = ax.bar(x - w / 2, naive_vals, w, label='Naive')
rects_kapra = ax.bar(x + w / 2, kapra_vals, w, label='KAPRA')

# 1. Generate appropriate labels
ylabel = 'Time (s)' \
    if metric == 'scalability' \
    else 'Loss'

title = dataset + ' - ' \
    + metric.capitalize().replace('_', ' ')

# 2. Add text for labels, title and custom ticks
ax.set_ylabel(ylabel)
ax.set_xlabel('# of records')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(rects_naive, ax)
autolabel(rects_kapra, ax)

fig.tight_layout()

# 3. Save stat figure to appropriate path
path_figure = Path(__file__).absolute().parent.parent / FIGURES_DIR
filename = 'Stat-' + dataset.rsplit('.', 1)[0] + '-' + metric

plt.savefig((path_figure / filename).with_suffix('.png'))
