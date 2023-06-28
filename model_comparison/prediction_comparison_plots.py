import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from matplotlib import pyplot as plt
import seaborn as sns
import os

engine = create_engine("postgresql+psycopg2://postgres@localhost/mas_ds")
conn = engine.connect()

hue_order = ['s_1632_100_8_811', 's_1632_100_16_811', 'm_1632_100_16_811', 'm_3408_150_16_811', 'm_4524_150_16_811',
             'm_6040_150_16_811', 'm_6040_150_16_721']

hue_order_class = ['Abwasser-eckig', 'Abwasser-rund', 'Abwasser-Einlaufschacht-eckig',
                   'Abwasser-Einlaufschacht-rund', 'andere-eckig', 'andere-rund']

colors = {'s_1632_100_8_811': '#1f77b4',
          's_1632_100_16_811': '#2ca02c',
          'm_1632_100_16_811': '#ff7f0e',
          's_3408_150_16_811': '#d62728',
          's_3408_150_64_811': '#9467bd',
          'm_3408_150_16_811': '#8c564b',
          'm_4524_150_16_811': '#e377c2',
          'm_6040_150_16_811': '#17becf',
          'm_6040_150_16_721': '#bcbd22'}


def plot_ture_predicted_point(conn, hue_order, colors):
    # plots point-truth
    query_statistic_point = text("SELECT * FROM public.v_test_prediction_statistic")
    test_predict_statistic = pd.read_sql_query(query_statistic_point, conn)

    # mean diff true predicted vs. orig
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_orig_count", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    # ax.set_yticks(np.arange(-15, 3, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean difference of predict vs. orig counts')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Differenz zwischen wahren und vohergesagten \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()


    # mean diff true predicted vs. orig (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_orig_count", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    ax.set_yticks(np.arange(-15, 3, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean difference of predict vs. orig counts')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Differenz zwischen wahren und vohergesagten \n Schachtpositionen (Score <= 0.05)')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean diff true predicted vs. false predicted
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_count", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    #ax.set_yticks(np.arange(-15, 15, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean count of false predicted')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittliche Anzahl falsch vohergesagter \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean diff true predicted vs. false predicted (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="diff_predict_05_count", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    #ax.set_yticks(np.arange(-15, 15, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean count of false predicted')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittliche Anzahl falsch vohergesagter \n Schachtpositionen (Score <= 0.05)')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean min-score
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="min_score", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean min score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittlich tiefster Score von korrekt vorhergesagten \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # min-score
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="min_score", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="min")
    ax.set_yticks(np.arange(0, 0.01, 0.001))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='min score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Tiefster Score von korrekt vorhergesagten \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean max-score
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="max_score", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean max score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittlich höchster Score von korrekt vorhergesagten \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # max-score
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="max_score", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="max")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='max score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Höchster Score von korrekt vorhergesagten \n Schachtpositionen')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean min-score (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="min_score_05", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean min score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittlich tiefster Score von korrekt vorhergesagten \n Schachtpositionen (Score <= 0.05)')
    # plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # min-score (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="min_score_05", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="min")
    ax.set_yticks(np.arange(0, 0.1, 0.01))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='min score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Tiefster Score von korrekt vorhergesagten \n Schachtpositionen (Score <= 0.05)')
    # plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # mean max-score (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="max_score_05", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="mean")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='mean max score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Durchschnittlich höchster Score von korrekt vorhergesagten \n Schachtpositionen (Score <= 0.05)')
    # plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # max-score (score >=0.05)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=test_predict_statistic, x="da_methode", y="max_score_05", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator="max")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='max score')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Höchster Score von korrekt vorhergesagten \n Schachtpositionen (Score <= 0.05)')
    # plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()


def plot_ture_predicted_class_orig_true(conn, hue_order, colors):
    # plots for class_truth - classes
    query_true_class = text("SELECT * FROM public.v_test_predict_statistic_class_orig_true")
    statistic_true_class = pd.read_sql_query(query_true_class, conn)

    sns.set_style('whitegrid')
    ax = sns.barplot(data=statistic_true_class, x="class_prediction", y="predict_true_count", hue="model",
                     hue_order=hue_order, palette=colors, errorbar=None, estimator=sum)
    # ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='sum of counts')
    plt.subplots_adjust(right=0.72, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Vorhergesagte Schachtdeckelklasse \n True vs. False')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()


def plot_ture_predicted_class_false(conn, hue_order, hue_order_class, colors):
    # read v_test_prediction_false_point
    query_false_class = text("SELECT * FROM public.v_test_prediction_false_point")
    statistic_false_class = pd.read_sql_query(query_false_class, conn)

    # plots for false class all models
    sns.set_style('whitegrid')
    ax = sns.countplot(data=statistic_false_class, x="class", order=['Abwasser-eckig', 'Abwasser-rund', 'Abwasser-Einlaufschacht-eckig',
                   'Abwasser-Einlaufschacht-rund', 'andere-eckig', 'andere-rund'], hue="model", hue_order=hue_order,
                       palette=colors)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.72, bottom=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Schachtdeckel pro Klasse')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = s_1632_100_8_811
    s_1632_100_8_811 = statistic_false_class[statistic_false_class["model"] == 's_1632_100_8_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=s_1632_100_8_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = s_1632_100_8_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = s_1632_100_16_811
    s_1632_100_16_811 = statistic_false_class[statistic_false_class["model"] == 's_1632_100_16_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=s_1632_100_16_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = s_1632_100_16_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = m_1632_100_16_811
    m_1632_100_16_811 = statistic_false_class[statistic_false_class["model"] == 'm_1632_100_16_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=m_1632_100_16_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = m_1632_100_16_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = m_3408_150_16_811
    m_3408_150_16_811 = statistic_false_class[statistic_false_class["model"] == 'm_3408_150_16_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=m_3408_150_16_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = m_3408_150_16_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = m_4524_150_16_811
    m_4524_150_16_811 = statistic_false_class[statistic_false_class["model"] == 'm_4524_150_16_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=m_4524_150_16_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = m_4524_150_16_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = m_6040_150_16_811
    m_6040_150_16_811 = statistic_false_class[statistic_false_class["model"] == 'm_6040_150_16_811'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=m_6040_150_16_811, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = m_6040_150_16_811')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()

    # plots for false class model = m_6040_150_16_721
    m_6040_150_16_721 = statistic_false_class[statistic_false_class["model"] == 'm_6040_150_16_721'].sort_values(by='da_methode')

    sns.set_style('whitegrid')
    ax = sns.countplot(data=m_6040_150_16_721, x="da_methode", hue="class", hue_order=hue_order_class)
    #ax.set_yticks(np.arange(-15, 2, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)
    ax.set(xlabel=None, ylabel='count false predicted')
    plt.subplots_adjust(right=0.68, bottom=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.title('Falsch vorhergesagte Klasse pro Data Augmenation Methode \n Modell = m_6040_150_16_721')
    #plt.savefig(r"D:\MAS_DataScience\plots\testval_percision_per_class.png", dpi=300)
    plt.show()
    plt.cla()



def plot_ture_predicted_class_false_best_models(conn, hue_order, colors):
    # plots for false class ('m_6040_150_16_811', 'm_6040_150_16_721')
    query_false_class = text("SELECT * FROM public.v_test_prediction_false_point "
                             "WHERE model IN('m_6040_150_16_811', 'm_6040_150_16_721')")
    statistic_false_class = pd.read_sql_query(query_false_class, conn)

    sns.set_style('whitegrid')
    ax = sns.countplot(data=statistic_false_class, x="da_methode", hue="model", palette=colors)
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
    ax = sns.countplot(data=statistic_false_class, x="class", hue="model", palette=colors)
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

