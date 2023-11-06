import tensorflow as tf
## The functional API
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_model(X):
    model_input = layers.Input(shape=X.shape[-1])
    x = layers.Dense(8,activation='relu')(model_input)
    x = layers.Dense(16,activation='relu')(x)
    model_output = layers.Dense(activation='linear')(x)
    
    return Model(model_input ,model_output)
