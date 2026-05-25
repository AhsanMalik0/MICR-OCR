import tensorflow as tf
from tensorflow.keras import layers
from train.DLOCRMICRSelfTrain.src.configuration import Config

class OCREncoder(tf.keras.Model):
    def __init__(self, config=None):
        super(OCREncoder, self).__init__()
        self.confg = config if config else Config()
        self.filters = self.confg.CONV_FILTERS
        self.kernel = self.confg.KERNEL_SIZE
        self.dropout = self.confg.DROPOUT_RATE
        
        # Block 1: 64 filters (cleaned up unused layers)
        self.conv1_1 = layers.Conv2D(self.filters[0], self.kernel, padding='same', kernel_initializer='he_normal')
        self.bn1_1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.SpatialDropout2D(self.dropout)
        
        # Block 2: 128 filters
        self.conv2_1 = layers.Conv2D(self.filters[1], self.kernel, padding='same', kernel_initializer='he_normal')
        self.bn2_1 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.SpatialDropout2D(self.dropout)
        
        # Block 3: 256 filters
        self.conv3_1 = layers.Conv2D(self.filters[2], self.kernel, padding='same', kernel_initializer='he_normal')
        self.bn3_1 = layers.BatchNormalization()
        
        # Block 4: 512 filters
        self.conv4_1 = layers.Conv2D(self.filters[3], self.kernel, padding='same', kernel_initializer='he_normal')
        self.bn4_2 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D((2, 1))  # Only pool height
        
        
        self.permute = layers.Permute((2, 1, 3))  # For sequence: width as time

    def build(self, input_shape):
        super(OCREncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        # Block 1
        x = self.conv1_1(inputs)
        x = layers.Activation('relu')(x)
        x = self.bn1_1(x, training=training)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2_1(x)
        x = layers.Activation('relu')(x)
        x = self.bn2_1(x, training=training)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3_1(x)
        x = layers.Activation('relu')(x)
        x = self.bn3_1(x, training=training)
        
        # Block 4
        x = self.conv4_1(x)
        x = self.bn4_2(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.pool4(x)

        
        # Reshape for sequence (width as time)
        x = self.permute(x)
        b, t, f, c = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, [b, t, f * c])
        return x
    
if __name__=="__main__":
    config = Config()
    model = OCREncoder()
    input_shape = (None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)  # Adjust based on your config
    model.build(input_shape)
    model.summary()