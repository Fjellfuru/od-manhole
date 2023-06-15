import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from matplotlib import pyplot as plt
import seaborn as sns
import os

engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
conn = engine.connect()
query = text("SELECT * FROM public.v_test_prediction_statistic")
test_predict_statistic = pd.read_sql_query(query, conn)


sns.set_style('whitegrid')
ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_orig_count", hue="model",
                 errorbar=None)
ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='predict_true_count_05')
plt.subplots_adjust(right=0.72, bottom=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vergleich Precision')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()

sns.set_style('whitegrid')
fig = plt.subplots(1, figsize=(10, 10))
ax = sns.boxplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_orig_count", hue="model", flierprops={"marker": "x"})
#ax.set_yticks(np.arange(-15, 2, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
ax.set_yticklabels(ax.get_yticklabels(), size=8)
ax.set(xlabel=None, ylabel='predict_true_count_05')
plt.subplots_adjust(right=0.8, bottom=0.2)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
plt.title('Vergleich Precision')
#plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
plt.show()
plt.cla()