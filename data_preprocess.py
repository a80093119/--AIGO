# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("training_data.csv")
print(data.head())
print(data.columns)
## drop na cols & row
data = data.drop(columns=["使用分區", "備註"])
data = data.dropna()
## replace "price" to log(price)
#data["單價"] = np.log(data["單價"])
data_train = data[['縣市', '土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '車位個數',
       '主建物面積', '陽台面積', '附屬建物面積']]

## 由於測試集與訓練集縣市總類不一致，因此需同時做dummy的處理 
predict_data = pd.read_csv("public_dataset.csv")
## drop na cols & row
predict_data = predict_data.drop(columns=["使用分區", "備註"])
predict_data = predict_data.dropna()
predict_data = predict_data[['縣市', '土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '車位個數',
       '主建物面積', '陽台面積', '附屬建物面積']]
## change 縣市 to dummy variable
total_data = pd.get_dummies(pd.concat([data_train, predict_data]))

## change 縣市 to dummy variable
data_train = total_data.iloc[:data_train.shape[0],:]
data_train_X = np.array(data_train).astype('float32')
data_train_y = np.array(data["單價"])

predict_data = total_data.iloc[data_train.shape[0]:,:]
predict_data_X = np.array(predict_data).astype('float32')
# In[]
"""
training model
"""

import tensorflow as tf
## The functional API
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_model(X):
    model_input = layers.Input(shape=X.shape[-1])
    x = layers.Dense(64,activation='relu')(model_input)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dense(64,activation='relu')(x)
    model_output = layers.Dense(1, activation='linear')(x)
    
    return Model(model_input ,model_output)

model = build_model(data_train_X)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(
    data_train_X, data_train_y,
    batch_size = 128,
    epochs = 500,
    validation_split = 0.2,
    callbacks=[callback]
    )

## model prediction
predict_y = model.predict(predict_data_X)
predict_y = np.exp(predict_y)

## save to csv
predict_csv = pd.read_csv("public_submission_template.csv")
predict_csv["predicted_price"] = predict_y
predict_csv.to_csv("predict_v1.csv", index=None)
# %%
