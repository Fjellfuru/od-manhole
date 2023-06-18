import model_comparison.model_comparison as mc

# plot metrics for each model
mc.compare_metrics_each_model()

# plot metric comparison over all models
all = mc.read_metrics_over_all_models()[0]
train_val_all = mc.create_df_train_val_metrics_over_all_models(all)
mc.compare_metrics_over_all_models(all, train_val_all, 'all')

# plot metric comparison from a part of the models
part = mc.read_metrics_over_all_models()[1]
train_val_part = mc.create_df_train_val_metrics_over_all_models(part)
mc.compare_metrics_over_all_models(part, train_val_part, 'part')