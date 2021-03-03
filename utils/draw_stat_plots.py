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

if __name__=="__main__":
    statfile_path = sys.argv[1]

    with open(Path(statfile_path), 'r') as f:
        # a. Parse the statfile header
        header = f.readline().split()

        metric = header[0]
        parameter = header[1]
        
        other_param_fixed = header[2]
        other_param_name = "P" if parameter=="K" else "K"

        # b. Parse the statfile rows
        lines = f.readlines()
        lines = list(map(lambda x: x.split(), lines))

    parameters = []

    datasets = {"facebook_microsoft" : [], "sales_transactions": [], "facebook_palestine": []}
    # k = None
    # check if data correct
    for i,l in enumerate(lines):
        if l[0] != "naive" and l[0] != "kapra":
            logger.error(f'Cannot interpret {l[0]} as a (k, P)-anonymity algorithm: only naive and KAPRA are supported')
            exit(1)
        if not l[1].isnumeric() or not l[2].isnumeric():
            logger.error("metric and parameter values should be numeric only")
            exit(1)

        if int(l[2]) not in parameters: parameters.append(int(l[2]))

        # keep track of which entries relate to which dataset
        if l[3] in datasets: datasets[l[3]].append(i) 
        else: 
            logger.error(f"unsupported dataset: {l[3]}")
            exit(1)

    fig, ax = plt.subplots()

    colors = ["blue", "green", "purple", "cyan", "orange", "pink"]
    for ds in datasets:
        param_naive_vals = [int(lines[i][2]) for i in datasets[ds] if lines[i][0] == "naive"]
        param_kapra_vals = [int(lines[i][2]) for i in datasets[ds] if lines[i][0] == "kapra"]
        naive_res_vals = [int(lines[i][1]) for i in datasets[ds] if lines[i][0] == "naive"]
        kapra_res_vals = [int(lines[i][1]) for i in datasets[ds] if lines[i][0] == "kapra"]

        ax.plot(param_naive_vals, naive_res_vals, label=f"{ds}.Naive", c=colors[-1])
        ax.scatter(param_naive_vals, naive_res_vals, c=colors[-1], marker="x")
        colors.pop(-1)

        ax.plot(param_kapra_vals, kapra_res_vals, label=f"{ds}.KAPRA", c=colors[-1])
        ax.scatter(param_kapra_vals, kapra_res_vals, c=colors[-1], marker="d")
        colors.pop(-1)

    # 1. Generate appropriate labels
    ylabel = 'Time (s)' \
        if metric == 'scalability' \
        else 'Loss'
        
    title = f"effects of tuning {parameter} on {metric.capitalize().replace('_', ' ')}, {other_param_name}={other_param_fixed if other_param_fixed else 0}"
    # 2. Add text for labels, title and custom ticks
    ax.set_ylabel(ylabel)
    ax.set_xlabel(parameter)
    ax.set_title(title)
    ax.set_xticks(parameters)  # ????
    ax.set_xticklabels(parameters)
    ax.legend()

    fig.tight_layout()

    # 3. Save stat figure to appropriate path
    path_figure = Path(__file__).absolute().parent.parent / FIGURES_DIR
    filename = f"Stat-{parameter}-{metric.capitalize().replace('_', '-')}"

    plt.savefig(path_figure / filename)
