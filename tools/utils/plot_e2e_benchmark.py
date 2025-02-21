import json

import matplotlib.pyplot as plt
import os
from typing import List
import numpy as np
import utils

colors = utils.colors

# Config:
INTERMEDIATE_PLOTS = False

y_label = ("latency (s)", "throughput (tokens/sec)") # "MFU" or "latency"
x_label = "selected adapters"

# name of the adapters
nm = ["1 LoRA", "2 LoRA", "1 Delta", "2 Delta", "1 LoRA 1 Delta"]
filenames = ['benchmark_1_lora_new.jsonl', 'benchmark_2_lora_new.jsonl', 'benchmark_1_delta_new.jsonl' , 'benchmark_2_delta_new.jsonl', 'benchmark_1_delta_1_lora_new.jsonl']

label_map = {"throughput (tokens/sec)": "output_throughput", "latency (s)": "mean_e2el_ms"}

title = (f"End to end latency",
            f"Output token throughput")
         
out_dir = "./"

data_names = ["total_token_throughput", "output_throughput", "mean_e2el_ms", "std_e2el_ms"]
# contains the results of the benchmark, the keys being the matrix sizes
data_metrics: dict[str, list[list]] = {}

for i, filename in enumerate(filenames):
    with open(filename, "r") as f:
        line = f.readline()
        data = json.loads(line.strip())
        data_metrics[nm[i]] = {}
        for data_name in data_names:
            if "ms" in data_name:
                data_metrics[nm[i]][data_name] = data["metrics"][data_name] / 1000
            else:
                data_metrics[nm[i]][data_name] = data["metrics"][data_name]

# create plot for each metric
plot_titles_and_labels = [
    (
        title[0],
        y_label[0],
        x_label,
    ),
    (
        title [1],
        y_label[1],
        x_label,
    )
]


def create_plot(
    title: str,
    x_label: str,
    y_label: str,
    names: list[str],
    metrics: dict[str, dict[str, list]],
):
    global colors

    # plot configurations
    plt.figure(figsize=(9, 5), dpi=300)
    plt.grid(True, which="major", axis="y", color="white", linestyle="-", linewidth=1.5, zorder=0)
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor("#dbdbdb")

    select = label_map[y_label]

    # plot data
    for j, name in enumerate(names):
        color = (0.7, 0.7, 0.7)
        if j == 2 or j == 3:
            color = colors[j-1]
        y_data = metrics[name][select]

        # Plot bars at measurement points
        plt.bar(name, y_data, color=color, label=name, zorder=2)
        if (select == "mean_e2el_ms"):
            # Compute confidence intervals
            std = metrics[name]["std_e2el_ms"]
            print(std)
            print(y_data)
            # Plot error bars at measurement points
            plt.errorbar(
                name,
                y_data,
                yerr=std,
                fmt="o",
                color=(0.4,0.4,0.4),
                ecolor=(0.4,0.4,0.4),
                elinewidth=1.5,
                capsize=4,
            )


    # add legend
    # plt.legend(names, fontsize=12)

    # remove border
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # set title and labels
    plt.title(title, fontsize = 14, loc="left", pad=24, fontweight="bold")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, rotation=0, labelpad=40, loc="bottom", fontsize=14)
    plt.gca().yaxis.set_label_coords(0, 1)
    plt.gca().get_yaxis().get_offset_text().set_x(-0.04)

    plt.savefig(
        f"{out_dir}/{select}.svg", format="svg", bbox_inches="tight"
    )
    plt.show()
    #plt.close()

for title, y_label, x_label in plot_titles_and_labels:
    create_plot(title, x_label, y_label, nm, data_metrics)
