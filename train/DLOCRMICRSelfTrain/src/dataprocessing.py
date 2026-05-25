from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os
from train.DLOCRMICRSelfTrain.src.configuration import Config


class DataProcessing:
    def __init__(self,  config=None):
        # self.confg = config
        self.confg = Config()

    def prepare_image(self, image):
        width, height = image.size
        scale = min(self.confg.IMG_WIDTH/width, self.confg.IMG_HEIGHT/height)
        if scale <1.0:
            new_width = int(width*scale)
            new_height = int(height*scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            width, height = new_width, new_height
        
        canvas = Image.new('L', (self.confg.IMG_WIDTH, self.confg.IMG_HEIGHT), 255)

        paste_x = (self.confg.IMG_WIDTH)
        paste_y = (self.confg.IMG_HEIGHT)
        canvas.paste(image, (paste_x, paste_y))
        image_array = np.array(canvas).astype(np.float32)
        image_array = image_array/255.0
        image_array = np.expand_dims(image_array, axis=-1)
        return image_array
    def process_image(self, image_path):
        image = Image.open(image_path).convert("L")
        image = self.prepare_image(image=image)
        return image
    def encoded_label(self, label_text, add_start=True, add_end=True):
        label_text = label_text.strip()
        encoded = []
        if add_start:
            encoded.append(self.confg.START_TOKEN)
        for char in label_text:
            if char in self.confg.char_to_num:
                encoded.append(self.confg.char_to_num[char])
            else:
                print(f"Warning: Character '{char}' not in vocabulary, skipping")
        if add_end:
            encoded.append(self.confg.END_TOKEN)
        if len(encoded) > self.confg.MAX_LABEL_LEN:
            encoded = encoded[:self.confg.MAX_LABEL_LEN-1] + [self.confg.END_TOKEN]
        return encoded
    
    def decode_label(self, token_ids, remove_special=True):
        decoded = []
        for token_id in token_ids:
            if token_id in [self.confg.START_TOKEN, self.confg.END_TOKEN, self.confg.PADDING_TOKEN]:
                continue
            elif token_id in self.confg.num_to_char:
                decoded.append(self.confg.num_to_char[token_id])
            else:
                print(f"Unknown Token> {token_id}")
        return ''.join(decoded)
    
    def prepare_data(self, datadir, verbose=True):
        image_files = [f for f in os.listdir(datadir) if f.lower().endswith(".jpg")]
        if len(image_files)==0:
            raise ValueError(f"No .jpg files fount in {datadir}")
        images = []
        labels = []
        skipped = 0
        for image_f in image_files:
            image_path = os.path.join(datadir, image_f)
            label_path = os.path.splitext(image_path)[0] + ".txt"
            if not os.path.exists(label_path):
                print(f"Warning Lable file not found for Image {image_f}, Skipped")
                skipped+=1
                continue
            try:
                image_array = self.process_image(image_path=image_path)
                with open(label_path, 'r') as f:
                    label_text = f.readline().strip()
                if not label_text:
                    print(f"Warning File Empty no Label Found for image {image_f}")
                    skipped+1
                    continue
                label_seq = self.encoded_label(label_text=label_text)
                if verbose:
                    print(f"loaded: {image_f}")
                    print(f"Label Text: {label_text}")
                    print(f"Label Tokens: {label_seq}")
                    print(f"Decoded: {self.decode_label(label_seq)}")
                
                images.append(image_array)
                labels.append(label_seq)
            
            except Exception as e:
                print(f"Error Processing {image_f}: as {str(e)}")
                skipped+=1
                continue
        if len(images)==0:
            raise ValueError(f"No Valid Images Loaded")
        
        padded_labels = tf.keras.preprocessing.sequence.pad_sequences(labels, 
                                                                        maxlen=self.confg.MAX_LABEL_LEN, 
                                                                        padding='post', 
                                                                        value=self.confg.PADDING_TOKEN)
        print(f"Skipped Images: {skipped}")
        print(f"Total Images: {len(image_files)}")
        print(f"Image Shape: {images[0].shape}") 
        print(f"Label Shape: {padded_labels.shape}")
        return np.array(images), np.array(padded_labels)
            

if __name__ == "__main__":
    config = Config()
    fun = DataProcessing(config=config)
    fun.prepare_data("/home/green-fin/pythonfiles/Prediction/test")