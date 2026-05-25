# trainer.py
import tensorflow as tf
import mlflow
import numpy as np
import os
import logging
from datetime import datetime
from train.DLOCRMICRSelfTrain.src.pipeline.metrix import MaskedLossAccuracy
from train.DLOCRMICRSelfTrain.src.pipeline.Evaluation import evaluate
from train.DLOCRMICRSelfTrain.src.models.Decoder import Decoder
from train.DLOCRMICRSelfTrain.src.models.Encoder import OCREncoder
from train.DLOCRMICRSelfTrain.src.configuration import Config
from train.DLOCRMICRSelfTrain.src.dataprocessing import DataProcessing


class MICRTrainer: 
    def __init__(self, encoder, decoder, optimizer, AccLoss, evaluation, config = None):
        super(MICRTrainer, self).__init__()
        self.config = config if config else Config()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.AccLoss = AccLoss
        self.evaluation = evaluation
        
        # Metrics tracking
        self.best_val_acc = 0.0  # Dynamically track best accuracy

        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.config.CHECKPOINT_DIR,
            max_to_keep=3
        )
        
        if self.manager.latest_checkpoint:
            print(f"Restoring from checkpoint: {self.manager.latest_checkpoint}")
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
        else:
            print("No checkpoint found, starting training from scratch.")
    
    @tf.function
    def train_step(self, images, labels):
        #Single training step
        batch_size = tf.shape(images)[0]
        
        with tf.GradientTape() as tape:
            features = self.encoder(images, training=True)
            hidden = self.decoder.initialize_hidden_state(batch_size)
            dec_input = tf.expand_dims(tf.fill([batch_size], self.config.START_TOKEN), 1)
            
            predictions, _, _ = self.decoder(dec_input, features, hidden, training=True)
            loss = self.AccLoss.masked_loss(labels, predictions)
        
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        acc = self.AccLoss.masked_accuracy(labels, predictions)
        return loss, acc
    
    def save_best_model(self, epoch, seq_acc):

        if self.best_val_acc is None or seq_acc > self.best_val_acc:
            saved_path = self.manager.save()
            print(f"New best accuracy ({seq_acc:.4f}). Saved checkpoint: {saved_path}")

            self.encoder.save_weights(os.path.join(self.config.MODEL_SAVE_DIR, "best_encoder.weights.h5"))
            self.decoder.save_weights(os.path.join(self.config.MODEL_SAVE_DIR, "best_decoder.weights.h5"))
            print("Saved encoder/decoder H5 weights.")

            self.best_val_acc = seq_acc

            return True

        return False

    def train(self, train_images, train_labels, val_images, val_labels):
        logging.basicConfig(level=logging.INFO)  
        logger = logging.getLogger(__name__)
        
        mlflow.set_experiment("MICR_OCR")
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log configuration
            mlflow.log_params(self.config.to_dict())
            
            best_epoch = 0
            patience_counter = 0
            
            for epoch in range(self.config.EPOCHS):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS}")
                logger.info(f"{'='*60}")
                
                # Shuffle training data
                indices = np.arange(len(train_images))
                np.random.shuffle(indices)
                train_images_shuffled = train_images#[indices]
                train_labels_shuffled = train_labels#[indices]
                
                # Training
                epoch_loss = 0
                epoch_acc = 0
                num_batches = 0
                
                for i in range(0, len(train_images), self.config.BATCH_SIZE):
                    batch_images = train_images_shuffled[i:i+self.config.BATCH_SIZE]
                    batch_labels = train_labels_shuffled[i:i+self.config.BATCH_SIZE]
                    
                    batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
                    batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
                    
                    loss, acc = self.train_step(batch_images, batch_labels)
                    
                    epoch_loss += loss.numpy()
                    epoch_acc += acc.numpy()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        print(f"Batch {num_batches} | Loss: {loss.numpy():.4f} | Acc: {acc.numpy():.4f}", end='\r')
                
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_acc / num_batches
                
                logger.info(f"\nTraining - Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("train_accuracy", avg_acc, step=epoch)
                
                # Validation
                char_acc, seq_acc, _p = self.evaluation(val_images, val_labels, self.encoder, self.decoder, self.config)
                logger.info(f"Validation - Char Acc: {char_acc*100:.2f}% | Seq Acc: {seq_acc*100:.2f}%")
                
                mlflow.log_metric("val_char_accuracy", char_acc, step=epoch)
                mlflow.log_metric("val_seq_accuracy", seq_acc, step=epoch)
                
                # Save best model if new accuracy is achieved
                if self.save_best_model(epoch, char_acc):
                    mlflow.log_metric("best_val_char_accuracy", char_acc)
                
                # Early stopping
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"\nEarly stopping at epoch {epoch+1}")
                    logger.info(f"Best validation accuracy: {self.best_val_acc*100:.2f}% at epoch {best_epoch+1}")
                    break
            
            # Log final model
            mlflow.tensorflow.log_model(self.encoder, artifact_path="encoder", registered_model_name="MICR_Encoder")
            mlflow.tensorflow.log_model(self.decoder, artifact_path="decoder", registered_model_name="MICR_Decoder")
            
            logger.info("\n✓ Training complete!")
            return self.best_val_acc

if __name__=="__main__":
    #
    config = Config()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.90
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    encoder = OCREncoder(config=config)
    decoder = Decoder(config=config)
    lossAcc = MaskedLossAccuracy(config=config)
    training = MICRTrainer(encoder=encoder, decoder=decoder, AccLoss=lossAcc, evaluation=evaluate, optimizer=optimizer, config=config)
    fun = DataProcessing(config=config)
    images, labels = fun.prepare_data("/home/green-fin/pythonfiles/Prediction/train/MICR")



    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    training.train(train_images=train_images, train_labels= train_labels, val_images=val_images, val_labels=val_labels)