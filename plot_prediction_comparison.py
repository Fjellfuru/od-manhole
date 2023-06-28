from sqlalchemy import create_engine, text
import model_comparison.prediction_comparison_plots as mcpr

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

mcpr.plot_ture_predicted_point(conn, hue_order, colors)
mcpr.plot_ture_predicted_class_orig_true(conn, hue_order, colors)
mcpr.plot_ture_predicted_class_false(conn, hue_order, hue_order_class, colors)