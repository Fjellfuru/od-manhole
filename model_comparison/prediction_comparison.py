import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

resulttest_yolov8s_1632_100_8_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8s_1632_100_8_811\test_val"
                                                 r"\predictions.json")
resulttest_yolov8s_1632_100_8_811['model'] = 's_1632_100_8_811'

resulttest_yolov8s_1632_100_16_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8s_1632_100_16_811\test_val"
                                                  r"\predictions.json")
resulttest_yolov8s_1632_100_16_811['model'] = 's_1632_100_16_811'

resulttest_yolov8m_1632_100_16_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8m_1632_100_16_811\test_val"
                                                  r"\predictions.json")
resulttest_yolov8m_1632_100_16_811['model'] = 'm_1632_100_16_811'

resulttest_yolov8m_3408_150_16_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8m_3408_150_16_811\test_val"
                                                  r"\predictions.json")
resulttest_yolov8m_3408_150_16_811['model'] = 'm_3408_150_16_811'

resulttest_yolov8m_4524_150_16_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8m_4524_150_16_811\test_val"
                                                  r"\predictions.json")
resulttest_yolov8m_4524_150_16_811['model'] = 'm_4524_150_16_811'

resulttest_yolov8m_6040_150_16_811 = pd.read_json(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_811\test_val"
                                                  r"\predictions.json")
resulttest_yolov8m_6040_150_16_811['model'] = 'm_6040_150_16_811'

resulttest_yolov8m_6040_150_16_721 = pd.read_json(r"D:\MAS_DataScience\training\yolov8m_6040_150_16_721\test_val"
                                                  r"\predictions.json")
resulttest_yolov8m_6040_150_16_721['model'] = 'm_6040_150_16_721'

df_result_testval = pd.concat(
    [resulttest_yolov8s_1632_100_8_811, resulttest_yolov8s_1632_100_16_811, resulttest_yolov8m_1632_100_16_811,
     resulttest_yolov8m_3408_150_16_811, resulttest_yolov8m_4524_150_16_811, resulttest_yolov8m_6040_150_16_811,
     resulttest_yolov8m_6040_150_16_721])

classes = {0: 'Abwasser-eckig', 1: 'Abwasser-rund', 2: 'Abwasser-Einlaufschacht-eckig',
           3: 'Abwasser-Einlaufschacht-rund', 4: 'andere-eckig', 5: 'andere-rund'}

df_result_testval['class'] = df_result_testval['category_id'].map(classes)

#add data_augmentation methode
df_result_testval['da_methode'] = None
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_sh'),
                                           'sharpen', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_gray'),
                                           'gray', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_bgr'),
                                           'bgr', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_grb'),
                                           'grb', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_cm1'),
                                           'contrast_decrease', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_cp2'),
                                           'contrast_increase', df_result_testval['da_methode'])
df_result_testval['da_methode'] = np.where(df_result_testval['image_id'].str.endswith('_escp2'),
                                           'contrast_increase_sharpen_emboss', df_result_testval['da_methode'])
df_result_testval['da_methode'] = df_result_testval['da_methode'].fillna('orig')

# add image_name
df_result_testval['image'] = None
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('_sh'),
                                      df_result_testval['image_id'].str.replace('_sh', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('_grey'),
                                      df_result_testval['image_id'].str.replace('_gray', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('bgr'),
                                      df_result_testval['image_id'].str.replace('_bgr', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('grb'),
                                      df_result_testval['image_id'].str.replace('_grb', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('_cm1'),
                                      df_result_testval['image_id'].str.replace('_cm1', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('_cp2'),
                                      df_result_testval['image_id'].str.replace('_cp2', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = np.where(df_result_testval['image_id'].str.endswith('_escp2'),
                                      df_result_testval['image_id'].str.replace('_escp2', '', regex=True),
                                      df_result_testval['image'])
df_result_testval['image'] = df_result_testval['image'].fillna(df_result_testval['image_id'])

df_result_testval['x'] = df_result_testval['bbox'].str[0]
df_result_testval['y'] = df_result_testval['bbox'].str[1]
df_result_testval['w'] = df_result_testval['bbox'].str[2]
df_result_testval['h'] = df_result_testval['bbox'].str[3]

print(df_result_testval.head(20).to_string())

df_result_group_image = df_result_testval.groupby(['image_id', 'model', 'da_methode'])[['score']]\
    .agg(['count', 'mean']).reset_index()

#print(df_result_group_image.to_string())

df_result_group_da_methode = df_result_testval.groupby(['class', 'model', 'da_methode'])\
    .agg(class_count=('score', 'count'), score_mean=('score', 'mean')).reset_index()

print(df_result_group_da_methode.info())

sns.set_style('whitegrid')
ax = sns.barplot(data=df_result_group_da_methode, x="class", y="score_mean", hue="da_methode", errorbar=None)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='mean score')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vergleich Precision')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

sns.catplot(data=df_result_testval, x="class", y="score", kind='boxen')
plt.show()
plt.cla()

df_result_group_model_da_methode = df_result_testval.groupby(['model', 'da_methode'])\
    .agg(class_count=('score', 'count'), score_mean=('score', 'mean')).reset_index()

df_result_group_model_da_methode_pivot = df_result_group_model_da_methode.pivot(index='model', columns='da_methode', values='score_mean')
sns.heatmap(df_result_group_model_da_methode_pivot, cmap="crest", annot=True, fmt=".1f")
plt.show()
plt.cla()