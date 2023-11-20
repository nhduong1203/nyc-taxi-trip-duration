import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import os


class FeatureEngineering:
    def __init__(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)

        self.train_data['log_trip_duration'] = np.log(self.train_data['trip_duration'].values + 1)

        return

    def all_feature_engineering(self):
        self.datetime_feature()
        self.binary_feature()
        self.distance_feature()


    def datetime_feature(self):
        self.train_data['pickup_datetime'] = pd.to_datetime(self.train_data.pickup_datetime)
        self.test_data['pickup_datetime'] = pd.to_datetime(self.test_data.pickup_datetime)
        self.train_data.loc[:, 'pickup_date'] = self.train_data['pickup_datetime'].dt.date
        self.test_data.loc[:, 'pickup_date'] = self.test_data['pickup_datetime'].dt.date
        self.train_data['dropoff_datetime'] = pd.to_datetime(self.train_data.dropoff_datetime)

        self.train_data.loc[:, 'pickup_weekday'] = self.train_data['pickup_datetime'].dt.weekday
        self.train_data.loc[:, 'pickup_hour'] = self.train_data['pickup_datetime'].dt.hour
        self.train_data.loc[:, 'pickup_minute'] = self.train_data['pickup_datetime'].dt.minute

        self.test_data.loc[:, 'pickup_weekday'] = self.test_data['pickup_datetime'].dt.weekday
        self.test_data.loc[:, 'pickup_hour'] = self.test_data['pickup_datetime'].dt.hour
        self.test_data.loc[:, 'pickup_minute'] = self.test_data['pickup_datetime'].dt.minute

        return

    def binary_feature(self):
        self.train_data['store_and_fwd_flag'] = 1 * (self.train_data.store_and_fwd_flag.values == 'Y')
        self.test_data['store_and_fwd_flag'] = 1 * (self.test_data.store_and_fwd_flag.values == 'Y')

        return

    def pca_coor(self):
        coords = np.vstack((self.train_data[['pickup_latitude', 'pickup_longitude']].values,
                            self.train_data[['dropoff_latitude', 'dropoff_longitude']].values,
                            self.test_data[['pickup_latitude', 'pickup_longitude']].values,
                            self.test_data[['dropoff_latitude', 'dropoff_longitude']].values))

        pca = PCA().fit(coords)
        self.train_data['pickup_pca0'] = pca.transform(self.train_data[['pickup_latitude', 'pickup_longitude']])[:, 0]
        self.train_data['pickup_pca1'] = pca.transform(self.train_data[['pickup_latitude', 'pickup_longitude']])[:, 1]
        self.train_data['dropoff_pca0'] = pca.transform(self.train_data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
        self.train_data['dropoff_pca1'] = pca.transform(self.train_data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
        self.test_data['pickup_pca0'] = pca.transform(self.test_data[['pickup_latitude', 'pickup_longitude']])[:, 0]
        self.test_data['pickup_pca1'] = pca.transform(self.test_data[['pickup_latitude', 'pickup_longitude']])[:, 1]
        self.test_data['dropoff_pca0'] = pca.transform(self.test_data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
        self.test_data['dropoff_pca1'] = pca.transform(self.test_data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

        return

    def haversine_array(self, lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def dummy_manhattan_distance(self, lat1, lng1, lat2, lng2):
        a = self.haversine_array(lat1, lng1, lat1, lng2)
        b = self.haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    def bearing_array(self, lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    def distance_feature(self):
        self.pca_coor()
        self.train_data.loc[:, 'distance_haversine'] = self.haversine_array(self.train_data['pickup_latitude'].values,
                                                             self.train_data['pickup_longitude'].values,
                                                             self.train_data['dropoff_latitude'].values,
                                                             self.train_data['dropoff_longitude'].values)
        self.train_data.loc[:, 'distance_dummy_manhattan'] = self.dummy_manhattan_distance(self.train_data['pickup_latitude'].values,
                                                                            self.train_data['pickup_longitude'].values,
                                                                            self.train_data['dropoff_latitude'].values,
                                                                            self.train_data['dropoff_longitude'].values)
        self.train_data.loc[:, 'direction'] = self.bearing_array(self.train_data['pickup_latitude'].values, self.train_data['pickup_longitude'].values,
                                                  self.train_data['dropoff_latitude'].values, self.train_data['dropoff_longitude'].values)
        self.train_data.loc[:, 'pca_manhattan'] = np.abs(self.train_data['dropoff_pca1'] - self.train_data['pickup_pca1']) + np.abs(self.train_data['dropoff_pca0'] - self.train_data['pickup_pca0'])

        self.test_data.loc[:, 'distance_haversine'] = self.haversine_array(self.test_data['pickup_latitude'].values,
                                                            self.test_data['pickup_longitude'].values,
                                                            self.test_data['dropoff_latitude'].values,
                                                            self.test_data['dropoff_longitude'].values)
        self.test_data.loc[:, 'distance_dummy_manhattan'] = self.dummy_manhattan_distance(self.test_data['pickup_latitude'].values,
                                                                           self.test_data['pickup_longitude'].values,
                                                                           self.test_data['dropoff_latitude'].values,
                                                                           self.test_data['dropoff_longitude'].values)
        self.test_data.loc[:, 'direction'] = self.bearing_array(self.test_data['pickup_latitude'].values, self.test_data['pickup_longitude'].values,
                                                 self.test_data['dropoff_latitude'].values, self.test_data['dropoff_longitude'].values)
        self.test_data.loc[:, 'pca_manhattan'] = np.abs(self.test_data['dropoff_pca1'] - self.test_data['pickup_pca1']) + np.abs(self.test_data['dropoff_pca0'] - self.test_data['pickup_pca0'])


if __name__ == "__main__":
    # print(os.listdir('../../'))
    fe = FeatureEngineering(train_file='../../data/train.csv', test_file='../../data/test.csv')
    print(fe.train_data.shape)
    print(fe.train_data.columns)
    fe.all_feature_engineering()
    print(fe.train_data.columns)
    print(fe.train_data.shape)


