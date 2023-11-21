from src import *
import numpy as np
import pandas as pd
from sklearn import linear_model


if __name__ == "__main__":
    train_data_path = "./data/train.csv"
    test_data_path = "./data/test.csv"

    dataset = MyDataset(train_data_path, test_data_path)
    dataset.load_data()
    train = dataset.get_train_data()
    test = dataset.get_test_data()

    # --- Visualize ---
    # visualization = Visualization(train, test)
    # visualization.plot_train_test_distribution()

    # --- Feature Engineering ---
    feature_engine = FeatureEngineering(train, test)
    feature_engine.all_feature_engineering()
    fe_train = feature_engine.get_train_data()
    fe_test = feature_engine.get_test_data()
    print(fe_train.columns)
    print(fe_test.columns)

    # --- Train model ---
    do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
                               'trip_duration', 'check_trip_duration',
                               'pickup_date', 'avg_speed_h', 'avg_speed_m',
                               'pickup_lat_bin', 'pickup_long_bin',
                               'center_lat_bin', 'center_long_bin',
                               'pickup_dt_bin', 'pickup_datetime_group']
    feature_names = [f for f in train.columns if f not in do_not_use_for_training]

    y = fe_train['log_trip_duration'].values
    filter_fe_train = fe_train[feature_names]
    filter_fe_test = fe_test[feature_names]

    model = linear_model.LinearRegression()
    trainer = TrainModel(model, filter_fe_train, y)
    trainer.train_and_evaluate()

