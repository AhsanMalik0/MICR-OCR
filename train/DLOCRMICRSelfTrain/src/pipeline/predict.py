# inference.py
import tensorflow as tf
import numpy as np
# from src.models.Encoder import OCREncoder
# from src.models.Decoder import Decoder

def inference(image, encoder, decoder, config):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)
    
    features = encoder(image, training=False)
    
    batch_size = tf.shape(image)[0]
    hidden = decoder.initialize_hidden_state(batch_size)
    dec_input = tf.expand_dims(tf.fill([batch_size], config.START_TOKEN), 1)
    
    predictions, _, attention_weights = decoder(dec_input, features, hidden, training=False)
    
    predicted_ids = tf.argmax(predictions, axis=-1).numpy()
    print(predicted_ids[0])
    return predicted_ids[0], predictions
