import tensorflow as tf
from tensorflow.keras import layers
from train.DLOCRMICRSelfTrain.src.configuration import Config
from train.DLOCRMICRSelfTrain.src.models.Attention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.confg = config if config else Config()
        
        self.units = self.confg.DECODER_UNITS
        self.dropout = self.confg.DROPOUT_RATE
        self.vocab_size = self.confg.vocab_size
        
        self.attention = BahdanauAttention(units=self.confg.ATTENTION_UNITS)
        self.fc_out = layers.Dense(self.vocab_size)
        self.act = layers.Activation('relu')

        self.rnn1 = layers.Bidirectional(layers.LSTM(self.units*2, return_sequences=True, dropout=self.dropout))
        self.rnn2 = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=self.dropout))
    def build(self, input_shape):
        super(Decoder, self).build(input_shape)
        
    def call(self, x, features, hidden, training=True):
        context_vector, attention_weights = self.attention(features, hidden)
        
        x = self.rnn1(context_vector)
        x = self.rnn2(x)
        
        x = self.fc_out(x)  # Shape:(Batch, Timestemp, vocab_size)
    
        state = None 
        return x, state, attention_weights
    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
if __name__=="__main__":
    config = Config()
    decoder = Decoder(config)
    decoder.summary()