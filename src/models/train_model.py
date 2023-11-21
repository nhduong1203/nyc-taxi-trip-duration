import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from evaluate import evaluate_metric  # Assuming you have a custom evaluation metric
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import linear_model
import pickle


class TrainModel:
    def __init__(self, model, train, y):
        self.model = model
        self.Xtr, self.Xv, self.ytr, self.yv = train_test_split(train, y, test_size=0.2, random_state=1987)

    def train_and_evaluate(self, params=None):
        if isinstance(self.model, XGBRegressor):
            start_time = time.time()
            dtrain = xgb.DMatrix(self.Xtr, label=self.ytr)
            dvalid = xgb.DMatrix(self.Xv, label=self.yv)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            self.model = xgb.train(params, dtrain, 100, watchlist, early_stopping_rounds=50,
                              maximize=False, verbose_eval=10)
            execution_time = time.time() - start_time
        else:
            start_time = time.time()
            self.model.fit(self.Xtr, self.ytr)
            yv_pred = self.model.predict(self.Xv)
            end_time = time.time()
            evaluation = evaluate_metric(self.yv, yv_pred)  # Replace with your evaluation function
            print(evaluation)
            print("Evaluation time: ", end_time - start_time)

        model_type = str(type(self.model)).split("'")[1].split('.')[-1]
        with open(f'./models/{model_type}.pkl', 'wb') as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":
    # Example usage
    train_data = pd.read_csv('./data/train.csv')
    train = train_data.drop(columns=['trip_duration', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag', 'trip_duration', 'id'])
    y = np.log(train_data['trip_duration'].values + 1)
    print(train_data.columns)
    print(y)
    model = linear_model.LinearRegression()

    trainer = TrainModel(model, train, y)
    trainer.train_and_evaluate()

    model_type= str(type(trainer.model)).split("'")[1].split('.')[-1]
    print(model)
    print(model_type)
