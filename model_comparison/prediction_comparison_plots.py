import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from matplotlib import pyplot as plt
import seaborn as sns
import os

engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
conn = engine.connect()

hue_order = ['m_1632_100_16_811', 'm_3408_150_16_811', 'm_4524_150_16_811',
             'm_6040_150_16_811', 'm_6040_150_16_721']

colors = {'s_1632_100_8_811': '#1f77b4',
          's_1632_100_16_811': '#2ca02c',
          'm_1632_100_16_811': '#ff7f0e',
          's_3408_150_16_811': '#d62728',
          's_3408_150_64_811': '#9467bd',
          'm_3408_150_16_811': '#8c564b',
          'm_4524_150_16_811': '#e377c2',
          'm_6040_150_16_811': '#17becf',
          'm_6040_150_16_721': '#bcbd22'}

# plots point-truth
query_statistic_point = text("SELECT * FROM public.v_test_prediction_statistic "
                             "WHERE model NOT IN ('s_1632_100_8_811', 's_1632_100_16_811')")
test_predict_statistic = pd.read_sql_query(query_statistic_point, conn)


sns.set_style('whitegrid')
ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_orig_count", hue="model",
                 hue_order=hue_order, palette=colors, errorbar=None)
ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='diff_predict_05_orig_count')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Differenz zwischen wahren und vohergesagten \n Schachtpositionen (Score <= 0.05')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

sns.set_style('whitegrid')
fig = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_orig_count", hue="model",
                 hue_order=hue_order, palette=colors, flierprops={"marker": "x"})
#ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='diff_predict_05_orig_count')
plt.subplots_adjust(right=0.8, bottom=0.2)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vergleich Precision')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

# plots for class_truth - classes
query_true_class = text("SELECT * FROM public.v_test_predict_statistic_class_orig_true "
                        "WHERE model NOT IN ('s_1632_100_8_811', 's_1632_100_16_811')")
statistic_true_class = pd.read_sql_query(query_true_class, conn)

sns.set_style('whitegrid')
ax = sns.barplot(data=statistic_true_class, x="class_prediction", y="predict_true_count", hue="model",
                 hue_order=hue_order, palette=colors, errorbar=None, estimator=sum)
# ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='diff_predict_05_orig_count')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vorhergesagten Schachtdeckelklass \n True oder False')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()


# plots for class_truth - only true or false
query_true_group_class = text("SELECT * FROM public.v_test_predict_statistic_class_true "
                              "WHERE model NOT IN ('s_1632_100_8_811', 's_1632_100_16_811')")
statistic_true_group_class = pd.read_sql_query(query_true_group_class, conn)

sns.set_style('whitegrid')
ax = sns.barplot(data=statistic_true_group_class, x="class_prediction", y="predict_true_count", hue="model",
                 hue_order=hue_order, palette=colors, errorbar=None, estimator=sum)
# ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='diff_predict_05_orig_count')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vergleich Vorhersage Klasse')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()


# plots for false class
query_false_class = text("SELECT * FROM public.v_test_prediction_false_point "
                         "WHERE model NOT IN ('s_1632_100_8_811', 's_1632_100_16_811')")
statistic_false_class = pd.read_sql_query(query_false_class, conn)

sns.set_style('whitegrid')
ax = sns.countplot(data=statistic_false_class, x="da_methode", hue="model", hue_order=hue_order, palette=colors)
#ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='count false predicted')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Falsch vorhergesagte Punkte pro Data Augmentation Methode')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

sns.set_style('whitegrid')
ax = sns.countplot(data=statistic_false_class, x="class", hue="model", hue_order=hue_order, palette=colors)
#ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='count false predicted')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Falsch vorhergesagte Punkte pro Klasse')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

sns.set_style('whitegrid')
ax = sns.countplot(data=statistic_false_class, x="class", hue="da_methode")
#ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='count false predicted')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Falsch vorhergesagte Punkte pro Data Augmentation Methode')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

