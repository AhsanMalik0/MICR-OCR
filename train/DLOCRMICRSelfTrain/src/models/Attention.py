import tensorflow as tf
from tensorflow.keras import layers
from train.DLOCRMICRSelfTrain.src.configuration import Config

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.units = units

        self.W1 = layers.Dense(self.units)
        self.W2 = layers.Dense(self.units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (B, T, 1)

        # Context for each time step (still (B, T, F))
        context_vector = attention_weights * features
        return context_vector, attention_weights
    
if __name__=="__main__":
    config = Config()
    attention = BahdanauAttention(units=256)