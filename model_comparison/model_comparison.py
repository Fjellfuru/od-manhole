import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from pathlib import Path

HOME = r"D:\MAS_DataScience\training"


def plot_results_overlay(result_csv: str, result_title: str, save_name: str):
    # read csv
    df = pd.read_csv(result_csv)
    df.columns = df.columns.str.lstrip()
    dest_path = Path(r"D:\MAS_DataScience\Dokumentation\plots\modellevaluation") / f"{save_name}{'.png'}"

    # generating subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6), tight_layout=True)
    fig.suptitle(result_title, fontsize=16, fontweight='bold')
    #plotting the dataframes
    df.plot(ax=axes[0][0], x='epoch', y=['train/box_loss', 'val/box_loss']).legend(prop={'size': 14})
    df.plot(ax=axes[0][1], x='epoch', y=['train/cls_loss', 'val/cls_loss']).legend(prop={'size': 14})
    df.plot(ax=axes[0][2], x='epoch', y=['train/dfl_loss', 'val/dfl_loss']).legend(prop={'size': 14})
    df.plot(ax=axes[1][0], x='epoch', y=['metrics/precision(B)', 'metrics/recall(B)']).legend(prop={'size': 14})
    df.plot(ax=axes[1][1], x='epoch', y=['metrics/mAP50(B)', 'metrics/mAP50-95(B)']).legend(prop={'size': 14})
    axes[1, 2].set_axis_off()
    plt.savefig(dest_path, dpi=300)
    #plt.show()
    plt.cla()


def compare_metrics_each_model():
    plot_results_overlay(os.path.join(HOME, 'yolov8s_1632_100_8_811/results.csv'),
                         'Metrics S-100-8 \n 1632 Bilder, Split 80/10/10', 's_1632_100_8_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8s_1632_100_16_811/results.csv'),
                         'Metrics S-100-16 \n 1632 Bilder, Split 80/10/10', 's_1632_100_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8m_1632_100_16_811/results.csv'),
                         'Metrics M-100-16 \n 1632 Bilder, Split 80/10/10', 'm_1632_100_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8s_3408_150_16_811/results.csv'),
                         'Matrics S-150-16 \n 3408 Bilder, Split 80/10/10', 's_3408_150_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8s_3408_150_64_811/results.csv'),
                         'Metrics S-150-64 \n 3408 Bilder, Split 8/1/1', 's_3408_150_64_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8m_3408_150_16_811/results.csv'),
                         'Metrics M-150-16 \n 3408 Bilder, Split 80/10/10', 'm_3408_150_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8m_4524_150_16_811/results.csv'),
                         'Metrics M-150-16 \n 4524 Bilder, Split 80/10/10', 'm_4524_150_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8m_6040_150_16_811/results.csv'),
                         'Metrics M-150-16 \n 6040 Bilder, Split 80/10/10', 'm_6040_150_16_811_metrics')
    plot_results_overlay(os.path.join(HOME, 'yolov8m_6040_150_16_721/results.csv'),
                         'Metrics M-150-16 \n 6040 Bilder, Split 70/20/10', 'm_6040_150_16_721_metrics')


def compare_metrics_over_all_models():
    yolov8s_1632_100_8_811 = pd.read_csv(os.path.join(HOME, 'yolov8s_1632_100_8_811/results.csv'))
    yolov8s_1632_100_8_811.columns = yolov8s_1632_100_8_811.columns.str.lstrip()
    yolov8s_1632_100_8_811['model'] = 's_1632_100_8_811'

    yolov8s_1632_100_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8s_1632_100_16_811/results.csv'))
    yolov8s_1632_100_16_811.columns = yolov8s_1632_100_16_811.columns.str.lstrip()
    yolov8s_1632_100_16_811['model'] = 's_1632_100_16_811'

    yolov8m_1632_100_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8m_1632_100_16_811/results.csv'))
    yolov8m_1632_100_16_811.columns = yolov8m_1632_100_16_811.columns.str.lstrip()
    yolov8m_1632_100_16_811['model'] = 'm_1632_100_16_811'

    yolov8m_3408_150_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8m_3408_150_16_811/results.csv'))
    yolov8m_3408_150_16_811.columns = yolov8m_3408_150_16_811.columns.str.lstrip()
    yolov8m_3408_150_16_811['model'] = 'm_3408_150_16_811'

    yolov8m_4524_150_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8m_4524_150_16_811/results.csv'))
    yolov8m_4524_150_16_811.columns = yolov8m_4524_150_16_811.columns.str.lstrip()
    yolov8m_4524_150_16_811['model'] = 'm_4524_150_16_811'

    yolov8m_6040_150_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8m_6040_150_16_811/results.csv'))
    yolov8m_6040_150_16_811.columns = yolov8m_6040_150_16_811.columns.str.lstrip()
    yolov8m_6040_150_16_811['model'] = 'm_6040_150_16_811'

    yolov8m_6040_150_16_721 = pd.read_csv(os.path.join(HOME, 'yolov8m_6040_150_16_721/results.csv'))
    yolov8m_6040_150_16_721.columns = yolov8m_6040_150_16_721.columns.str.lstrip()
    yolov8m_6040_150_16_721['model'] = 'm_6040_150_16_721'

    df_model_comp = pd.concat([yolov8s_1632_100_8_811, yolov8s_1632_100_16_811, yolov8m_1632_100_16_811,
                               yolov8m_3408_150_16_811, yolov8m_4524_150_16_811, yolov8m_6040_150_16_811,
                               yolov8m_6040_150_16_721])

    # Create Data Frame for train and val comparison
    df_model_comp_train = df_model_comp.loc[:, ['epoch', 'model', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss']]
    df_model_comp_train['task'] = 'train'
    df_model_comp_train.rename(columns={'train/box_loss': 'box_loss', 'train/cls_loss': 'cls_loss',
                                        'train/dfl_loss': 'dfl_loss'}, inplace=True)

    df_model_comp_val = df_model_comp.loc[:, ['epoch', 'model', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']]
    df_model_comp_val['task'] = 'val'
    df_model_comp_val.rename(
        columns={'val/box_loss': 'box_loss', 'val/cls_loss': 'cls_loss', 'val/dfl_loss': 'dfl_loss'},
        inplace=True)

    df_model_comp_train_val = pd.concat([df_model_comp_train, df_model_comp_val])

    # Comparison mAP50
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp, x="epoch", y="metrics/mAP50(B)", hue="model", lw=1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='mAP50')
    plt.legend(loc='lower right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich mAP50')
    plt.savefig(r"D:\MAS_DataScience\Test\training_map50_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison mAP50-95
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp, x="epoch", y="metrics/mAP50-95(B)", hue="model", lw=1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='mAP50')
    plt.legend(loc='lower right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich mAP50-95')
    plt.savefig(r"D:\MAS_DataScience\Test\training_map50-95_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison Recall
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp, x="epoch", y="metrics/recall(B)", hue="model", lw=1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='mAP50')
    plt.legend(loc='lower right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Recall')
    plt.savefig(r"D:\MAS_DataScience\Test\training_recall_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison Precision
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp, x="epoch", y="metrics/precision(B)", hue="model", lw=1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='mAP50')
    plt.legend(loc='lower right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Precision')
    plt.savefig(r"D:\MAS_DataScience\Test\training_precision_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison Box-Loss
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp_train_val, x="epoch", y="box_loss", hue="model", style='task', lw=1)
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='box loss')
    plt.legend(loc='upper right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Box-Loss')
    plt.savefig(r"D:\MAS_DataScience\Test\training_box_loss_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison Class-Loss
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp_train_val, x="epoch", y="cls_loss", hue="model", style='task', lw=1)
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='cls loss')
    plt.legend(loc='upper right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Class-Loss')
    plt.savefig(r"D:\MAS_DataScience\Test\training_cls_loss_epoch.png", dpi=300)
    plt.show()
    plt.cla()

    # Comparison DFL-Loss
    sns.set_style('whitegrid')
    ax = sns.lineplot(data=df_model_comp_train_val, x="epoch", y="dfl_loss", hue="model", style='task', lw=1)
    ax.set_xticklabels(ax.get_xticklabels(), size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel='Epoche', ylabel='box loss')
    plt.legend(loc='upper right', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Box-Loss')
    plt.savefig(r"D:\MAS_DataScience\Test\training_dfl_loss_epoch.png", dpi=300)
    plt.show()
    plt.cla()


def compare_test_val():

    metrics_yolov8s_1632_100_8_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8s_1632_100_8_811\test_val"
                                                 r"\metrics_per_class.csv", sep=';')
    metrics_yolov8s_1632_100_8_811['model'] = 's_1632_100_8_811'

    metrics_yolov8s_1632_100_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8s_1632_100_16_811\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8s_1632_100_16_811['model'] = 's_1632_100_16_811'

    metrics_yolov8m_1632_100_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_1632_100_16_811\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8m_1632_100_16_811['model'] = 'm_1632_100_16_811'

    metrics_yolov8m_3408_150_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_3408_150_16_811\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8m_3408_150_16_811['model'] = 'm_3408_150_16_811'

    metrics_yolov8m_4524_150_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_4524_150_16_811\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8m_4524_150_16_811['model'] = 'm_4524_150_16_811'

    metrics_yolov8m_6040_150_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_811\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8m_6040_150_16_811['model'] = 'm_6040_150_16_811'

    metrics_yolov8m_6040_150_16_721 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_721\test_val"
                                                  r"\metrics_per_class.csv", sep=';')
    metrics_yolov8m_6040_150_16_721['model'] = 'm_6040_150_16_721'

    df_model_testval = pd.concat(
      [metrics_yolov8s_1632_100_8_811, metrics_yolov8s_1632_100_16_811, metrics_yolov8m_1632_100_16_811,
       metrics_yolov8m_3408_150_16_811, metrics_yolov8m_4524_150_16_811, metrics_yolov8m_6040_150_16_811,
       metrics_yolov8m_6040_150_16_721])

    # Precision
    sns.set_style('whitegrid')
    ax = sns.barplot(data=df_model_testval, x="Class", y="Box_P", hue="model")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='Precision')
    plt.subplots_adjust(right=0.72, bottom=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Precision')
    plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # Recall
    sns.set_style('whitegrid')
    ax = sns.barplot(data=df_model_testval, x="Class", y="Box_R", hue="model")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='Recall')
    plt.subplots_adjust(right=0.72, bottom=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Vergleich Recall')
    plt.savefig(r"D:\MAS_DataScience\plots\testval_recall_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # Box_mAP50
    sns.set_style('whitegrid')
    ax = sns.barplot(data=df_model_testval, x="Class", y="Box_mAP50", hue="model")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='Recall')
    plt.subplots_adjust(right=0.72, bottom=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Vergleich mAP50')
    plt.savefig(r"D:\MAS_DataScience\plots\testval_map50_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # Box_mAP50-95
    sns.set_style('whitegrid')
    ax = sns.barplot(data=df_model_testval, x="Class", y="Box_mAP50-95", hue="model")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='Recall')
    plt.subplots_adjust(right=0.72, bottom=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Vergleich mAP50-95')
    plt.savefig(r"D:\MAS_DataScience\plots\testval_map50-95_per_class.png", dpi=300)
    plt.show()
    plt.cla()