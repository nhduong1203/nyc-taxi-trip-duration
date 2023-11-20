import pandas as pd
import os
import numpy as np


class MyDataset:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = None
        self.test_data = None

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_file)
            self.test_data = pd.read_csv(self.test_file)
            print("Data loaded successfully!")
        except FileNotFoundError:
            print("File not found. Please check the file paths.")

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_test_info(self):
        if self.train_data is not None:
            self.train_data.info()
        else:
            print("Train data not loaded yet.")

        if self.test_data is not None:
            self.test_data.info()
        else:
            print("Test data not loaded yet.")

        return

    def check_duplicate_data(self):
        print(f'Target: {set(self.train_data.columns) - set(self.test_data.columns)}')
        print('Id is unique.') if self.train_data.id.nunique() == self.train_data.shape[0] else print('oops')
        print('Train and test sets are distinct.') if len(
            np.intersect1d(self.train_data.id.values, self.test_data.id.values)) == 0 else print('oops')
        print('No missing values.') if self.train_data.count().min() == self.train_data.shape[
            0] and self.test_data.count().min() == self.test_data.shape[0] else print('oops')
        print('The store_and_fwd_flag has only two values {}.'.format(
            str(set(self.train_data.store_and_fwd_flag.unique()) | set(self.test_data.store_and_fwd_flag.unique()))))
        return


if __name__ == "__main__":
    # print(os.listdir('../../'))
    data = MyDataset(train_file='../../data/train.csv', test_file='../../data/test.csv')
    data.load_data()
    train = data.get_train_data()
    test = data.get_train_data()
    data.get_train_test_info()
    data.check_duplicate_data()