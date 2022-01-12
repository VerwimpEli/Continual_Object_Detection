import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datasets as ds

plt.style.use('seaborn')


def main():
    file = './results/test/1/roi_out.csv'

    roi_preds = pd.read_csv(file)
    roi_preds = roi_preds.groupby('class')

    mean_preds = {}

    for name, group in roi_preds:
        predictions = group.iloc[:, 6:].to_numpy()
        mean_preds[name] = np.mean(predictions, axis=0)

    fig, axes = plt.subplots(4, 5)
    x = np.arange(len(roi_preds) + 1)

    for i, ax in enumerate(axes.flatten()):
        ax.bar(x, mean_preds[i])
        ax.set_title(ds.VOC_CAT_SHORT_NAMES[i])
        ax.set_ylim((-4, 14))

    plt.show()


if __name__ == '__main__':
    main()























