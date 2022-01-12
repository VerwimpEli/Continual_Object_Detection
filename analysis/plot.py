import argparse
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')


def _select_rows_with_index_str(results: pd.DataFrame, index_str: str):
    # Returns all rows that have index_str in their index
    return results.loc[results.index.to_series().str.contains(index_str)]


def plot_map(results: pd.DataFrame, tasks: Sequence[Sequence[int]]):
    map_rows = _select_rows_with_index_str(results, 'mAP')
    result_arr = map_rows.sort_index().to_numpy()

    for i, task in enumerate(tasks):
        task_ap = np.mean(result_arr[:, task], axis=1)
        plt.plot(task_ap, label=f'Task {i}')
        print(task_ap)

    plt.xticks(range(len(tasks)), [f'T{i}' for i in range(len(tasks))])
    plt.ylabel("Task AP")
    plt.legend()


def plot_confusions(results: pd.DataFrame, tasks: Sequence[Sequence[int]]):
    nclasses = len(results.columns)
    ntasks = len(tasks)

    conf_rows = _select_rows_with_index_str(results, 'conf')
    result_arr = conf_rows.to_numpy()
    gts = results.loc['num_gt'].to_numpy().reshape((1, -1))

    fig, axes = plt.subplots(1, ntasks, figsize=(2 + 7*ntasks, 7))

    for i in range(ntasks):
        confusion = result_arr[i*nclasses:(i+1)*nclasses]
        confusion /= gts
        sns.heatmap(confusion, fmt='.0f', ax=axes[i], annot=True, cmap='Reds', vmin=0.0, vmax=100)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('result_file')
    args = parser.parse_args()

    results = pd.read_csv(args.result_file)
    results = results.set_index('Name')

    tasks = [range(0, 10), range(11, 20)]

    plot_map(results, tasks)
    plot_confusions(results, tasks)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
