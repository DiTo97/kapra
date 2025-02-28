"""
Draw linechart plots to compare naive and KAPRA (k, P)-anonymity implementations tuning against some metrics on progressively larger datasets.

An input statfile is expected to conform to this header-format:
<metric> <fixed_param> <fixed_param_value>

where <metric> can either be one of: eta, avg_pattern_loss, tot_pattern_loss, avg_value_loss, or tot_value_loss;
where <fixed_param> can either be k or P, and represents the parameter to vary the other one against in (k, P)-anonymity tuning.

An input statfile is expected to conform to this row-format:
<algorithm> <value> <tuned_param_value> <dataset>

where <value> should adhere to the specified <metric>;
where <tuned_param_value> should adhere to either P or k, that is, the parameter that's actively being tuned.
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
    statfile_path = 'C:/Users/gvlos/Documents/GitHub/kapra/results/K16_tot_pattern_loss.txt'

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
        # if not l[1].isnumeric() or not l[2].isnumeric():
        #     logger.error("metric and parameter values should be numeric only")
        #     exit(1)

        if int(l[2]) not in parameters: parameters.append(int(l[2]))

        # keep track of which entries relate to which dataset
        if l[3] in datasets: datasets[l[3]].append(i) 
        else: 
            logger.error(f"unsupported dataset: {l[3]}")
            exit(1)

    fig, ax = plt.subplots()

    colors = ["pink", "cyan", "purple", "green", "orange", "blue"]
    for ds in datasets:
        param_naive_vals = [float(lines[i][2]) for i in datasets[ds] if lines[i][0] == "naive"]
        param_kapra_vals = [float(lines[i][2]) for i in datasets[ds] if lines[i][0] == "kapra"]
        naive_res_vals = [float(lines[i][1]) for i in datasets[ds] if lines[i][0] == "naive"]
        kapra_res_vals = [float(lines[i][1]) for i in datasets[ds] if lines[i][0] == "kapra"]

        ax.plot(param_naive_vals, naive_res_vals, label=f"{ds}.Naive", ls='--', c=colors[-1])
        ax.scatter(param_naive_vals, naive_res_vals, c=colors[-1], marker="x")
        #colors.pop(-1)

        ax.plot(param_kapra_vals, kapra_res_vals, label=f"{ds}.KAPRA", c=colors[-1])
        ax.scatter(param_kapra_vals, kapra_res_vals, c=colors[-1], marker="d")
        colors.pop(-1)

    # 1. Generate appropriate labels
    if metric == 'tot_value_loss':
        ylabel = 'Total Value loss'
    elif metric == 'avg_value_loss':
        ylabel = 'Average value loss'
    elif metric == 'tot_pattern_loss':
        ylabel = 'Total Pattern loss'
    elif metric == 'avg_pattern_loss':
        ylabel = 'Average Pattern loss'
    elif metric == 'eta':
        ylabel = 'Execution time (s)'
        
    title = f"Tuning {parameter} on {ylabel}, ({other_param_name}={other_param_fixed})"
    # 2. Add text for labels, title and custom ticks
    ax.set_ylabel(ylabel)
    ax.set_xlabel(parameter)
    ax.set_title(title)
    ax.set_xticks(parameters)  # ????
    ax.set_xticklabels(parameters)
    ax.legend(loc='best', fontsize=7)

    fig.tight_layout()

    # 3. Save stat figure to appropriate path
    path_figure = Path(__file__).absolute().parent.parent / FIGURES_DIR
    filename = f"Stat-{parameter}-{metric.capitalize().replace('_', '-')}"

    plt.savefig(path_figure / filename, dpi=200)
