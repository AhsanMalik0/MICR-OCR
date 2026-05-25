import tensorflow as tf
from train.DLOCRMICRSelfTrain.src.configuration import Config

class MaskedLossAccuracy:
    def __init__(self, config: Config):
        self.config = config
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def masked_loss(self, y_true, y_pred):
        loss = self.loss_object(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, self.config.PADDING_TOKEN), loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def masked_accuracy(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, self.config.PADDING_TOKEN), tf.float32)
        pred_ids = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        correct = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
        correct *= mask
        return tf.reduce_sum(correct) / tf.reduce_sum(mask)
    
    def ppo_loss(self, old_probs, new_probs, advantage, epsilon=0.2):
        #
        ratio = new_probs / old_probs
        #
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        loss = tf.minimum(ratio * advantage, clipped_ratio * advantage)
        return -tf.reduce_mean(loss)
    def calculate_reward(labels, predictions):
        #Calculate Rewards for RL Agent learning
        # 
        correct = tf.equal(tf.argmax(predictions, axis=-1), labels)
        reward = tf.reduce_mean(tf.cast(correct, tf.float32))
        return reward
    

if __name__=="__main__":
    config = Config()
    masked = MaskedLossAccuracy(config=config)
