# from train.DLOCRMICRSelfTrain.src.models.Encoder import OCREncoder
# from train.DLOCRMICRSelfTrain.src.models.Attention import BahdanauAttention
# from train.DLOCRMICRSelfTrain.src.models.Decoder import Decoder
from train.DLOCRMICRSelfTrain.src.pipeline.trainer import Decoder, OCREncoder, CONFIG
from train.DLOCRMICRSelfTrain.src.configuration import Config
from train.DLOCRMICRSelfTrain.src.pipeline.predict import inference
from train.DLOCRMICRSelfTrain.src.dataprocessing import DataProcessing
import tensorflow as tf
import os
from PIL import Image, ImageTk
import numpy as np
import sys
print(sys.executable)
class InfHead:
    def __init__(self):
        self.config = Config()
        self.processing = DataProcessing(config=self.config)
        self.encoder = OCREncoder(config=CONFIG)
        self.decoder = Decoder(config=CONFIG)
        self.encoder.build(input_shape=(None, None, None, None))  
        self.decoder.build(input_shape=(None, None, None))
        self._load_weights()

    def _load_weights(self):
        
        checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        if os.path.exists(self.config.CHECKPOINT_DIR):
            manager = tf.train.CheckpointManager(
                checkpoint,
                self.config.CHECKPOINT_DIR,
                max_to_keep=3
            )
            if manager.latest_checkpoint:
                checkpoint.restore(manager.latest_checkpoint).expect_partial()
                print(f"Restored checkpoint from {manager.latest_checkpoint}")
                # Check the model weights after loading
                print("Encoder Weights:", self.encoder.trainable_variables)
                print("Decoder Weights:", self.decoder.trainable_variables)
            else:
                print("No checkpoint found in directory.")
        else:
            print(f"Checkpoint directory not found: {self.config.CHECKPOINT_DIR}")
        self.encoder.summary()

    def prediction(self, image):
        image = self.processing.prepare_image(image=image)
        image = np.array(image)
        predicted_ids, predictions = inference(image=image, encoder=self.encoder, decoder=self.decoder, config=self.config)
        predicted_text = self.processing.decode_label(token_ids=predicted_ids)
        return predicted_text
    
if __name__ == "__main__":
    pred = InfHead()
    config = Config()
    image_path = input("Image Path>> ")
    image = Image.open(image_path).convert("L")
    prediction = pred.prediction(image)
    print(f"Prediction> {prediction}")