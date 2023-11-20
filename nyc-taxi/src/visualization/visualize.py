# src/visualization/visualize.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# from data.dataset import MyDataset


class Visualization:
    def __init__(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)

    # def get_train_info(self):
    #     if self.train is not None:
    #         return self.train.info()
    #     else:
    #         print("Train data not loaded yet.")
    #
    # def get_test_info(self):
    #     if self.test is not None:
    #         return self.test.info()
    #     else:
    #         print("Test data not loaded yet.")

    # def plot_target(self):
    #     plt.hist(np.log(self.train_data['trip_duration'].values+1), bins=100)
    #     plt.xlabel('log(trip_duration)')
    #     plt.ylabel('number of train records')
    #     plt.show()
    #     return

    def plot_train_test_distribution(self):
        features = self.test_data.columns
        print(features)

        fig = plt.figure(figsize=(10, 30))
        # list_hist_columns = ['vendor_id', 'passenger_count', 'pickup_datetime', 'store_and_fwd_flag']
        # vendor_id, passenger_count

        column_name = 'vendor_id'
        plt.subplot(4, 2, 1)
        self.train_data[column_name].value_counts().sort_index().plot(kind='bar', color='green', label='Train')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(4, 2, 2)
        self.test_data[column_name].value_counts().sort_index().plot(kind='bar', color='blue', label='Test')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        column_name = 'passenger_count'
        plt.subplot(4, 2, 3)
        self.train_data[column_name].value_counts().sort_index().plot(kind='bar', color='green', label='Train')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(4, 2, 4)
        self.test_data[column_name].value_counts().sort_index().plot(kind='bar', color='blue', label='Test')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        column_name = 'store_and_fwd_flag'
        plt.subplot(4, 2, 5)
        self.train_data[column_name].value_counts().sort_index().plot(kind='bar', color='green', label='Train')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(4, 2, 6)
        self.test_data[column_name].value_counts().sort_index().plot(kind='bar', color='blue', label='Test')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()

        N = 100000
        city_long_border = (-74.03, -73.75)
        city_lat_border = (40.63, 40.85)
        plt.subplot(4, 2, 7)
        plt.scatter(self.train_data['pickup_longitude'].values[:N], self.train_data['pickup_latitude'].values[:N],
                color='green', s=1, label='train', alpha=0.1)
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.ylim(city_lat_border)
        plt.xlim(city_long_border)

        plt.subplot(4, 2, 8)
        plt.scatter(self.test_data['pickup_longitude'].values[:N], self.test_data['pickup_latitude'].values[:N],
                    color='blue', s=1, label='train', alpha=0.1)
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.ylim(city_lat_border)
        plt.xlim(city_long_border)

        plt.tight_layout()
        plt.show()

        return

    # def plot_target_correlation(self):


if __name__ == "__main__":

    visualize = Visualization(train_file='../../data/train.csv', test_file='../../data/test.csv')
    visualize.plot_train_test_distribution()

