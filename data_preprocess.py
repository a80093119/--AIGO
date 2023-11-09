# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import outside_fuction as out_func

# In[]
data = pd.read_csv("training_data.csv")
print(data.head())
print(data.columns)
## drop na cols & row
data = data.drop(columns=["使用分區", "備註"])
data = data.dropna()
## replace "price" to log(price)
#data["單價"] = np.log(data["單價"])
data_train = data[['縣市', '鄉鎮市區', "主要用途", "主要建材", "建物型態", '土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '車位個數', '橫坐標', '縱坐標',
       '主建物面積', '陽台面積', '附屬建物面積']]
## numeric columns
data.describe()
## catagory columns
catagory_col = ["縣市", "鄉鎮市區", "路名", "主要用途", "主要建材", "建物型態"]
data[catagory_col].describe()

## 由於測試集與訓練集縣市總類不一致，因此需合併同時做dummy的處理 
predict_data = pd.read_csv("public_dataset.csv")
## drop na cols & row
predict_data = predict_data.drop(columns=["使用分區", "備註"])
predict_data = predict_data.dropna()
predict_data = predict_data[['縣市', '鄉鎮市區', "主要用途", "主要建材", "建物型態", '土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '車位個數', '橫坐標', '縱坐標',
       '主建物面積', '陽台面積', '附屬建物面積']]

## combind train and public data
total_data = pd.concat([data_train, predict_data])
# In[]
"""
新增欄位並透過座標轉換計算距離，並根據縣市等級給予加權轉換成分數，分數越高越好，沒有則為-1分
"""
total_data["lat"] = total_data.apply(lambda x: out_func.twd97_to_lonlat(x["橫坐標"], x["縱坐標"])[1], axis=1)
total_data["lng"] = total_data.apply(lambda x: out_func.twd97_to_lonlat(x["橫坐標"], x["縱坐標"])[0], axis=1)

## 火車站點資料
TW_train_data = pd.read_csv("./external_data/火車站點資料.csv")
TW_train_data = TW_train_data[TW_train_data["車站級別"] <= 2]
from geopy.distance import geodesic
train_dist_list = []
for i in range(total_data.shape[0]):
    # print(i)
    town = total_data.iloc[i,:]
    house_dist_list = []
    for j in range(TW_train_data.shape[0]):
        house_dist_list.append((TW_train_data.iloc[j]["車站級別"]+1) * geodesic((town["lat"], town["lng"]), (TW_train_data.iloc[j]["lat"], TW_train_data.iloc[j]["lng"])).km)
    train_dist_list.append(min(house_dist_list))
print(train_dist_list)
total_data["train_station_dist"] = train_dist_list
total_data.to_csv("train_X_data.csv", index=None)

## 大學資料
university_data = pd.read_csv("./external_data/大學基本資料.csv")
university_data = university_data.groupby("學校名稱", as_index=False).agg({'總計': np.sum, "lat": np.mean, "lng": np.mean})
plt.boxplot(university_data["總計"])
university_data = university_data[university_data["總計"] >= university_data["總計"].mean()]
unversity_dist_list = []
for i in range(total_data.shape[0]):
    # print(i)
    town = total_data.iloc[i,:]
    house_unversity_dist_list = []
    for j in range(university_data.shape[0]):
        house_unversity_dist_list.append(geodesic((town["lat"], town["lng"]), (university_data.iloc[j]["lat"], university_data.iloc[j]["lng"])).km)
    unversity_dist_list.append(min(house_unversity_dist_list))
print(unversity_dist_list)
total_data["university_dist"] = unversity_dist_list
total_data.to_csv("train_X_data.csv", index=None)
# In[]
## 鄉鎮市區修改dummy做法，因會發散，改成使用Freq_encoding，同時連上縣市避免地名重複問題
total_data["地區"] = total_data.agg('{0[縣市]}{0[鄉鎮市區]}'.format, axis=1)
for key, value in total_data["地區"].value_counts().to_dict().items():
    total_data["地區"][total_data["地區"] == key] = value
total_data["地區"] = total_data["地區"].astype("int")
## 其餘category variables使用dummies 
total_data = total_data.drop(columns=["鄉鎮市區", "橫坐標", "縱坐標"])
total_data = pd.get_dummies(total_data)
## split train and predict
data_train = total_data.iloc[:data_train.shape[0],:]
data_train_X = np.array(data_train).astype('float32')
data_train_y = np.array(data["單價"])

predict_data = total_data.iloc[data_train.shape[0]:,:]
predict_data_X = np.array(predict_data).astype('float32')

# In[]
"""
training model with machine learning
"""
# machine learning
## XGB
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(data_train_X, data_train_y, test_size=0.2)

xgboostModel = xgb.XGBRegressor(n_estimators=500, learning_rate= 0.03)
# 使用訓練資料訓練模型
xgboostModel.fit(data_train_X, data_train_y)
# 預測成功的比例
print('訓練集: ',xgboostModel.score(X_train,y_train))
print('測試集: ',xgboostModel.score(X_test,y_test))
print(xgboostModel.feature_importances_)
# plot
plt.bar(range(len(xgboostModel.feature_importances_)), xgboostModel.feature_importances_)
plt.show()
# 使用Public資料預測分類
predicted = xgboostModel.predict(predict_data_X)
## save to csv
predict_csv = pd.read_csv("public_submission_template.csv")
predict_csv["predicted_price"] = predicted
predict_csv.to_csv("predict_xgb_v2.csv", index=None)

## KNN
from sklearn.neighbors import KNeighborsRegressor
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=5)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))

# In[]
"""
training model with deep learning
"""

import tensorflow as tf
## The functional API
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_model(X):
    model_input = layers.Input(shape=X.shape[-1])
    x = layers.BatchNormalization()
    x = layers.Dense(64,activation='LeakyReLU')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128,activation='LeakyReLU')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,activation='LeakyReLU')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,activation='LeakyReLU')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512,activation='LeakyReLU')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,activation='LeakyReLU')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,activation='LeakyReLU')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128,activation='LeakyReLU')(x)
    model_output = layers.Dense(1, activation='linear')(x)
    
    return Model(model_input ,model_output)

model = build_model(data_train_X)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.MeanAbsolutePercentageError(),
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
model.summary()
checkpoint_filepath = "./tmp/checkpoint"
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_freq="epoch",
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(
    data_train_X, data_train_y,
    batch_size = 128,
    epochs = 10000,
    validation_split = 0.2,
    callbacks=[model_checkpoint_callback],
    shuffle = True
    )

# The model weights (that are considered the best) are loaded into the
# model.
model.load_weights(checkpoint_filepath)

## find worst data
predict_train_y = model.predict(data_train_X)
data_false_preidct_X = data[abs(predict_train_y.reshape(-1) - data_train_y) >= 0.5]
data_false_preidct_X.describe()
## catagory columns
catagory_col = ["縣市", "鄉鎮市區", "路名", "主要用途", "主要建材", "建物型態"]
data_false_preidct_X[catagory_col].describe()

## model prediction
predict_y = model.predict(predict_data_X)
predict_y = predict_y

## save to csv
predict_csv = pd.read_csv("public_submission_template.csv")
predict_csv["predicted_price"] = predict_y
predict_csv.to_csv("predict_v1.csv", index=None)
# In[]
## add XGB result to DL model
predicted = xgboostModel.predict(data_train_X)
## split train and predict
data_train["xgb_predict"] = predicted
data_train_X = np.array(data_train).astype('float32')

model = build_model(data_train_X)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.MeanAbsolutePercentageError(),
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
model.summary()
checkpoint_filepath = './tmp/checkpoint'
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_freq="epoch",
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(
    data_train_X, data_train_y,
    batch_size = 128,
    epochs = 10000,
    validation_split = 0.2,
    callbacks=[model_checkpoint_callback]
    )

## add XGB result to DL model
predicted = xgboostModel.predict(predict_data_X)
predict_data = total_data.iloc[data_train.shape[0]:,:]
predict_data["xgb_predict"] = predicted
predict_data_X = np.array(predict_data).astype('float32')
## model prediction
predict_y = model.predict(predict_data_X)
predict_y = predict_y

## save to csv
predict_csv = pd.read_csv("public_submission_template.csv")
predict_csv["predicted_price"] = predict_y
predict_csv.to_csv("predict_v2.csv", index=None)
# %%
target_encode() 