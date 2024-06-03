# %%
from dotenv import load_dotenv
import pandas as pd
import os
import sys
import datetime

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    

load_dotenv()

# %%
def df_to_X_y(df: pd.DataFrame):
    df['days_on_shelf'] = pd.to_datetime(df['date']) - pd.to_datetime(df['release_date'])
    df['days_on_shelf'] = df['days_on_shelf'].dt.days.astype('int16')
    cols = df.columns.tolist()
    cols = cols[:4] + cols[-1:] + cols[4:-1]
    df = df[cols]
    df.pop('date')
    df.pop('release_date')

    df_as_np = df.to_numpy()
    X = df_as_np[:, 4:]
    y = df_as_np[:, :4]
    return X, y

def date_range(start: datetime.date, end: datetime.date, step: datetime.timedelta):
    while start < end:
        yield start
        start += step

# %%
df_train = pd.concat([
    pd.read_csv(f'../datasets/{d.strftime("%Y-%m-%d")}.csv') for d in date_range(
        datetime.date(2019, 9, 30), 
        datetime.date(2020, 1, 6),
        datetime.timedelta(weeks=1),
    )
])
df_val = pd.concat([
    pd.read_csv(f'../datasets/{d.strftime("%Y-%m-%d")}.csv') for d in date_range(
        datetime.date(2020, 1, 6), 
        datetime.date(2020, 2, 3),
        datetime.timedelta(weeks=1),
    )
])
df_test = pd.concat([
    pd.read_csv(f'../datasets/{d.strftime("%Y-%m-%d")}.csv') for d in date_range(
        datetime.date(2020, 2, 3), 
        datetime.date(2020, 2, 25),
        datetime.timedelta(weeks=1),
    )
])

X_train, y_train = df_to_X_y(df_train)
X_val, y_val = df_to_X_y(df_val)
X_test, y_test = df_to_X_y(df_test)

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

# %%
import tensorflow as tf
tf.__version__, tf.config.list_physical_devices('GPU')

# %%
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam


model = Sequential()
model.add(InputLayer((22, 1)))
model.add(LSTM(64))
model.add(Dense(8, 'relu'))
model.add(Dense(4, 'linear'))

model.summary()

# %%
cp = ModelCheckpoint('../model/model1.keras', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

# %%
p = model.predict(X_test)
df = pd.DataFrame({ 
    'predict_view': p[:, 0],
    'view': y_test[:, 0],
    'predict_cart': p[:, 1],
    'cart': y_test[:, 1],
    'predict_remove_from_cart': p[:, 2],
    'remove_from_cart': y_test[:, 2],
    'predict_purchase': p[:, 3],
    'purchase': y_test[:, 3],
})
df.to_csv('test.csv')


