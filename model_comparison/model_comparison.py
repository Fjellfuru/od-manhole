import pandas as pd
from matplotlib import pyplot as plt
import os

HOME = r"D:\MAS_DataScience\training"

def plot_results_overlay(result_csv: str, result_title: str):
  # read csv
  df = pd.read_csv(result_csv)
  df.columns = df.columns.str.lstrip()

  # generating subplots
  fig,axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6), tight_layout=True)
  fig.suptitle(result_title, fontsize=16)
  #plotting the dataframes
  df.plot(ax=axes[0][0], x='epoch', y=['train/box_loss', 'val/box_loss'])
  df.plot(ax=axes[0][1], x='epoch', y=['train/cls_loss', 'val/cls_loss'])
  df.plot(ax=axes[0][2], x='epoch', y=['train/dfl_loss', 'val/dfl_loss'])
  df.plot(ax=axes[1][0], x='epoch', y=['metrics/precision(B)', 'metrics/recall(B)'])
  df.plot(ax=axes[1][1], x='epoch', y=['metrics/mAP50(B)', 'metrics/mAP50-95(B)'])
  # displaying the title
  plt.show()


# plot_results_overlay(os.path.join(HOME, 'yolov8s_1632_100_8/results.csv'), 'S-100-8')
# plot_results_overlay(os.path.join(HOME, 'yolov8s_1632_100_16/results.csv'), 'S-100-16')
# plot_results_overlay(os.path.join(HOME, 'yolov8m_1632_100_16/results.csv'), 'M-100-16')
# plot_results_overlay(os.path.join(HOME, 'yolov8s_3408_150_16/results.csv'), 'S-150-16')
# plot_results_overlay(os.path.join(HOME, 'yolov8m_3408_150_16/results.csv'), 'M-150-16')
# plot_results_overlay(os.path.join(HOME, 'yolov8s_3408_150_64/results.csv'), 'S-150-64')
# plot_results_overlay(os.path.join(HOME, 'yolov8m_4524_150_16/results.csv'), 'M-150-16 4524 Images')
plot_results_overlay(os.path.join(HOME, 'yolov8m_6040_150_16_811/results.csv'), 'M-150-16 6040 Images Split 811')
plot_results_overlay(os.path.join(HOME, 'yolov8m_6040_150_16_721/results.csv'), 'M-150-16 6040 Images Split 721')

yolov8s_3408_150_16 = pd.read_csv(os.path.join(HOME, 'yolov8s_3408_150_16/results.csv'))
yolov8s_3408_150_16.columns = yolov8s_3408_150_16.columns.str.lstrip()

yolov8m_3408_150_16 = pd.read_csv(os.path.join(HOME, 'yolov8m_3408_150_16/results.csv'))
yolov8m_3408_150_16.columns = yolov8m_3408_150_16.columns.str.lstrip()

yolov8s_3408_150_64 = pd.read_csv(os.path.join(HOME, 'yolov8s_3408_150_64/results.csv'))
yolov8s_3408_150_64.columns = yolov8s_3408_150_64.columns.str.lstrip()

yolov8m_4524_150_16 = pd.read_csv(os.path.join(HOME, 'yolov8m_4524_150_16/results.csv'))
yolov8m_4524_150_16.columns = yolov8m_4524_150_16.columns.str.lstrip()

yolov8m_6040_150_16_811 = pd.read_csv(os.path.join(HOME, 'yolov8m_6040_150_16_811/results.csv'))
yolov8m_6040_150_16_811.columns = yolov8m_6040_150_16_811.columns.str.lstrip()

yolov8m_6040_150_16_721 = pd.read_csv(os.path.join(HOME, 'yolov8m_6040_150_16_721/results.csv'))
yolov8m_6040_150_16_721.columns = yolov8m_6040_150_16_721.columns.str.lstrip()

yolov8s_3408_150_16['metrics/mAP50(B)'].plot(label='yolov8s_3408_150_16', color='orange')
yolov8m_3408_150_16['metrics/mAP50(B)'].plot(label='yolov8m_3408_150_16', color='blue')
yolov8s_3408_150_64['metrics/mAP50(B)'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['metrics/mAP50(B)'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['metrics/mAP50(B)'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['metrics/mAP50(B)'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')

# adding title to the plot
plt.title('Vergleich mAP50')

# adding Label to the x-axis
plt.xlabel('epoch')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()


yolov8s_3408_150_16['metrics/mAP50-95(B)'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['metrics/mAP50-95(B)'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['metrics/mAP50-95(B)'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['metrics/mAP50-95(B)'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['metrics/mAP50-95(B)'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['metrics/mAP50-95(B)'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')

# adding title to the plot
plt.title('Vergleich mAP50-95')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()

yolov8s_3408_150_16['metrics/recall(B)'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['metrics/recall(B)'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['metrics/recall(B)'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['metrics/recall(B)'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['metrics/recall(B)'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['metrics/recall(B)'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')

# adding title to the plot
plt.title('Vergleich Recall')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()

yolov8s_3408_150_16['metrics/precision(B)'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['metrics/precision(B)'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['metrics/precision(B)'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['metrics/precision(B)'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['metrics/precision(B)'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['metrics/precision(B)'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')

# adding title to the plot
plt.title('Vergleich Precision')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()

yolov8s_3408_150_16['val/dfl_loss'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['val/dfl_loss'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['val/dfl_loss'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['val/dfl_loss'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['val/dfl_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['val/dfl_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')
yolov8s_3408_150_16['train/dfl_loss'].plot(label='yolov8s_3408_150_16', color='blue', linestyle='dashed')
yolov8m_3408_150_16['train/dfl_loss'].plot(label='yolov8m_3408_150_16', color='orange', linestyle='dashed')
yolov8s_3408_150_64['train/dfl_loss'].plot(label='yolov8s_3408_150_64', color='cyan', linestyle='dashed')
yolov8m_4524_150_16['train/dfl_loss'].plot(label='yolov8m_4524_150_16', color='red', linestyle='dashed')
yolov8m_6040_150_16_811['train/dfl_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta', linestyle='dashed')
yolov8m_6040_150_16_721['train/dfl_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet', linestyle='dashed')


# adding title to the plot
plt.title('Vergleich DFL-Loss')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()

yolov8s_3408_150_16['val/cls_loss'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['val/cls_loss'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['val/cls_loss'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['val/cls_loss'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['val/cls_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['val/cls_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')
yolov8s_3408_150_16['train/cls_loss'].plot(label='yolov8s_3408_150_16', color='blue', linestyle='dashed')
yolov8m_3408_150_16['train/cls_loss'].plot(label='yolov8m_3408_150_16', color='orange', linestyle='dashed')
yolov8s_3408_150_64['train/cls_loss'].plot(label='yolov8s_3408_150_64', color='cyan', linestyle='dashed')
yolov8m_4524_150_16['train/cls_loss'].plot(label='yolov8m_4524_150_16', color='red', linestyle='dashed')
yolov8m_6040_150_16_811['train/cls_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta', linestyle='dashed')
yolov8m_6040_150_16_721['train/cls_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet', linestyle='dashed')

# adding title to the plot
plt.title('Vergleich CLS-Loss')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()

yolov8s_3408_150_16['val/box_loss'].plot(label='yolov8s_3408_150_16', color='blue')
yolov8m_3408_150_16['val/box_loss'].plot(label='yolov8m_3408_150_16', color='orange')
yolov8s_3408_150_64['val/box_loss'].plot(label='yolov8s_3408_150_64', color='cyan')
yolov8m_4524_150_16['val/box_loss'].plot(label='yolov8m_4524_150_16', color='red')
yolov8m_6040_150_16_811['val/box_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta')
yolov8m_6040_150_16_721['val/box_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet')
yolov8s_3408_150_16['train/box_loss'].plot(label='yolov8s_3408_150_16', color='blue', linestyle='dashed')
yolov8m_3408_150_16['train/box_loss'].plot(label='yolov8m_3408_150_16', color='orange', linestyle='dashed')
yolov8s_3408_150_64['train/box_loss'].plot(label='yolov8s_3408_150_64', color='cyan', linestyle='dashed')
yolov8m_4524_150_16['train/box_loss'].plot(label='yolov8m_4524_150_16', color='red', linestyle='dashed')
yolov8m_6040_150_16_811['train/box_loss'].plot(label='yolov8m_6040_150_16_811', color='magenta', linestyle='dashed')
yolov8m_6040_150_16_721['train/box_loss'].plot(label='yolov8m_6040_150_16_721', color='darkviolet', linestyle='dashed')


# adding title to the plot
plt.title('Vergleich Box-Loss')

# adding Label to the x-axis
plt.xlabel('Epoche')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()


metrics_yolov8s_1632_100_8 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8s_1632_100_8\test_val\metrics_per_class.csv")
metrics_yolov8s_1632_100_16 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8s_1632_100_16\test_val\metrics_per_class.csv")
metrics_yolov8m_1632_100_16 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_1632_100_16\test_val\metrics_per_class.csv")
metrics_yolov8m_3408_150_16 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_3408_150_16\test_val\metrics_per_class.csv")
metrics_yolov8m_4524_150_16 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_4524_150_16\test_val\metrics_per_class.csv")
metrics_yolov8m_6040_150_16_811 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_811\test_val\metrics_per_class.csv")
metrics_yolov8m_6040_150_16_721 = pd.read_csv(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_721\test_val\metrics_per_class.csv")


metrics_yolov8s_1632_100_8['Box_P'].plot(label='yolov8s_3408_150_16', color='blue')


# adding title to the plot
plt.title('Vergleich Box-Loss')

# adding legend to the curve
plt.legend()
plt.show()
plt.cla()