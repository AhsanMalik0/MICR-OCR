import os
import string
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from keras import regularizers
import time
import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image
from IPython.display import display, HTML
import cv2
import pytesseract
from ultralytics import YOLO 
import re



char_list = string.digits + 'OTDB?'  # 0-9, A, D, O, T, space
char_to_num = {char: i+3 for i, char in enumerate(char_list)}  # Start from 3

# Special tokens
START_TOKEN = 0
END_TOKEN = 1
PADDING_TOKEN = 2

# Add special tokens to conversion
num_to_char = {i+3: char for char, i in zip(char_list, range(len(char_list)))}
num_to_char[START_TOKEN] = '<START>'
num_to_char[END_TOKEN] = '<END>'
num_to_char[PADDING_TOKEN] = '<PAD>'

# Vocabulary size includes all characters + special tokens
num_classes = len(char_list) + 3  # chars + START + END + PAD

# Image and sequence parameters
TARGET_HEIGHT = 64      
TARGET_WIDTH = 256      
MAX_LABEL_LEN = 64      
print(f"Vocabulary size: {num_classes}")
print(f"Character mapping: {char_to_num}")
print(f"Image size: {TARGET_HEIGHT}x{TARGET_WIDTH}")

# ============= IMAGE PREPROCESSING =============
def prepare_image(image, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):
    width, height = image.size
    
    # Calculate scaling factor to fit within target while preserving aspect ratio
    scale = min(target_width / width, target_height / height)
    
    if scale < 1.0:  # Only resize if image is larger
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        width, height = new_width, new_height
    
    # Create a white canvas of target size
    canvas = Image.new('L', (target_width, target_height), 255)
    
    # Paste the image in the center
    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2
    canvas.paste(image, (paste_x, paste_y))
    
    return canvas
# ============= Single IMAGE PREPROCESSING =============

def preprocess_image(image_path, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):

    # Load image
    image = Image.open(image_path).convert("L")
    
    # Resize/pad
    image = prepare_image(image, target_height, target_width)
    
    # Convert to numpy and normalize
    image_np = np.array(image).astype(np.float32)
    
    # Normalize to [0, 1]
    image_np = image_np / 255.0
    
    # Optional: Invert if text is dark on light background
    # image_np = 1.0 - image_np
    
    # Add channel dimension: (H, W, 1)
    image_np = np.expand_dims(image_np, axis=-1)
    
    return image_np
def preprocess_array(image, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):

    # Load image
    image = Image.fromarray(image, mode='L')
    # image = Image.fromarray(image.astype('uint8'), mode='L')
    
    # Resize/pad
    image = prepare_image(image, target_height, target_width)
    
    # Convert to numpy and normalize
    image_np = np.array(image).astype(np.float32)
    
    # Normalize to [0, 1]
    image_np = image_np / 255.0
    
    # Optional: Invert if text is dark on light background
    # image_np = 1.0 - image_np
    
    # Add channel dimension: (H, W, 1)
    image_np = np.expand_dims(image_np, axis=-1)
    
    return image_np
# ============= LABEL ENCODING =============
def encode_label(label_text, max_len=MAX_LABEL_LEN, 
                 add_start=True, add_end=True):

    # Clean the text (remove extra spaces, etc.)
    label_text = label_text.strip()
    
    # Encode characters
    encoded = []
    
    if add_start:
        encoded.append(START_TOKEN)
    
    for char in label_text:
        if char in char_to_num:
            encoded.append(char_to_num[char])
        else:
            print(f"Warning: Character '{char}' not in vocabulary, skipping")
    
    if add_end:
        encoded.append(END_TOKEN)
    
    # Truncate if too long
    if len(encoded) > max_len:
        print(f"Warning: Label too long ({len(encoded)} > {max_len}), truncating")
        encoded = encoded[:max_len-1] + [END_TOKEN]
    
    return encoded

def decode_label(token_ids, remove_special=True):
    decoded = []
    
    for token_id in token_ids:
        if remove_special and token_id in [START_TOKEN, END_TOKEN, PADDING_TOKEN]:
            continue
        
        if token_id in num_to_char:
            decoded.append(num_to_char[token_id])
    
    return ''.join(decoded)

# ============= DATASET PREPARATION =============
def prepare_data(data_dir, img_height=TARGET_HEIGHT, img_width=TARGET_WIDTH, 
                 max_label_len=MAX_LABEL_LEN, verbose=True):
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    
    if len(image_files) == 0:
        raise ValueError(f"No .jpg files found in {data_dir}")
    
    images = []
    labels = []
    skipped = 0
    
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.splitext(img_path)[0] + ".txt"
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_file}, skipping")
            skipped += 1
            continue
        
        try:
            # Load and preprocess image
            image_np = preprocess_image(img_path, img_height, img_width)
            
            # Load label
            with open(label_path, "r") as f:
                label_text = f.readline().strip()
            
            # Skip empty labels
            if not label_text:
                print(f"Warning: Empty label for {img_file}, skipping")
                skipped += 1
                continue
            
            # Encode label (with START and END tokens)
            label_seq = encode_label(label_text, max_len=max_label_len)
            
            if verbose:
                print(f"Loaded: {img_file}")
                print(f"  Label text: '{label_text}'")
                print(f"  Label tokens: {label_seq}")
                print(f"  Decoded: '{decode_label(label_seq)}'")
            
            images.append(image_np)
            labels.append(label_seq)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            skipped += 1
            continue
    
    if len(images) == 0:
        raise ValueError("No valid images loaded!")
    
    # Pad labels to max length
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
        labels, 
        maxlen=max_label_len, 
        padding='post', 
        value=PADDING_TOKEN
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset prepared:")
    print(f"  Total files: {len(image_files)}")
    print(f"  Successfully loaded: {len(images)}")
    print(f"  Skipped: {skipped}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Label shape: {padded_labels.shape}")
    print(f"  Label range: [{padded_labels.min()}, {padded_labels.max()}]")
    print(f"{'='*60}\n")
    
    return np.array(images), np.array(padded_labels)


# ============= DATA VALIDATION =============
def validate_dataset(images, labels):

    print("Validating dataset...")
    
    # Check shapes
    assert len(images) == len(labels), "Mismatch between images and labels"
    assert images.ndim == 4, f"Images should be 4D, got {images.ndim}D"
    assert labels.ndim == 2, f"Labels should be 2D, got {labels.ndim}D"
    
    # Check value ranges
    assert images.min() >= 0 and images.max() <= 1, \
        f"Images should be normalized to [0,1], got [{images.min()}, {images.max()}]"
    
    assert labels.min() >= 0 and labels.max() < num_classes, \
        f"Labels should be in [0, {num_classes}), got [{labels.min()}, {labels.max()}]"
    
    # Check for valid sequences
    for i, label in enumerate(labels[:10]):  # Check first 10
        # Remove padding
        label_no_pad = label[label != PADDING_TOKEN]
        
        # Check if has START and END
        if len(label_no_pad) > 0:
            if label_no_pad[0] != START_TOKEN:
                print(f"Warning: Label {i} doesn't start with START_TOKEN")
            
            # Find first END token
            end_indices = np.where(label_no_pad == END_TOKEN)[0]
            if len(end_indices) == 0:
                print(f"Warning: Label {i} doesn't have END_TOKEN")
    
    # Print statistics
    label_lengths = np.sum(labels != PADDING_TOKEN, axis=1)
    print(f"\nLabel length statistics:")
    print(f"  Min: {label_lengths.min()}")
    print(f"  Max: {label_lengths.max()}")
    print(f"  Mean: {label_lengths.mean():.2f}")
    print(f"  Median: {np.median(label_lengths):.2f}")
    
    print("\nSample labels (first 5):")
    for i in range(min(5, len(labels))):
        label_no_pad = labels[i][labels[i] != PADDING_TOKEN]
        decoded = decode_label(label_no_pad)
        print(f"  {i}: {label_no_pad} -> '{decoded}'")
    
    print("\n✓ Dataset validation passed!")
# ============= CONFIGURATION =============
CONFIG = {
    # Model Architecture
    'conv_filters': [64, 128, 256, 512],
    'kernel_size': (3, 3),
    'embedding_dim': 256,
    'decoder_units': 128,
    'attention_units': 512,
    'dropout_rate': 0.5,
    'vocab_size': num_classes,

    
    # Image preprocessing
    'img_height': TARGET_HEIGHT,
    'img_width': TARGET_WIDTH,
    'normalize': True,
    # Model 
    'batch_size': 2,
    'epochs': 80,
    'initial_lr': 0.001,
    'gradient_clip': 4.0,
    'teacher_forcing_decay': 0.02,
}

class OCREncoder(tf.keras.Model):
    def __init__(self, config=None):
        super(OCREncoder, self).__init__()
        self.confg = config if config else Config()
        filters = self.confg["conv_filters"]
        kernel = self.confg["kernel_size"]
        dropout = self.confg["dropout_rate"]
        
        # Block 1: 64 filters (cleaned up unused layers)
        self.conv1_1 = layers.Conv2D(filters[0], kernel, padding='same', kernel_initializer='he_normal')
        self.bn1_1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.SpatialDropout2D(dropout)
        
        # Block 2: 128 filters
        self.conv2_1 = layers.Conv2D(filters[1], kernel, padding='same', kernel_initializer='he_normal')
        self.bn2_1 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.SpatialDropout2D(dropout)
        
        # Block 3: 256 filters
        self.conv3_1 = layers.Conv2D(filters[2], kernel, padding='same', kernel_initializer='he_normal')
        self.bn3_1 = layers.BatchNormalization()
        
        # Block 4: 512 filters
        self.conv4_1 = layers.Conv2D(filters[3], kernel, padding='same', kernel_initializer='he_normal')
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

class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.confg = config if config else Config()
        
        self.units = self.confg['decoder_units']
        self.dropout = self.confg['dropout_rate']
        self.vocab_size = self.confg['vocab_size']
        
        self.attention = BahdanauAttention(units=self.confg['attention_units'])
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
# Initialize models
encoder = OCREncoder(CONFIG)
decoder = Decoder(CONFIG)
encoder.build(input_shape=(None, None, None, None))  
decoder.build(input_shape=(None, None, None))

# decoder.summary()
def masked_accuracy(real, pred):
    # real: [batch, seq_len], pred: [batch, seq_len, vocab_size]
    mask = tf.cast(tf.math.not_equal(real, 0), tf.float32)
    pred_ids = tf.argmax(pred, axis=-1, output_type=real.dtype)  # shape: [batch, seq_len]
    correct = tf.cast(tf.equal(real, pred_ids), tf.float32)
    correct *= mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def masked_loss(y_true, y_pred, padding_token):
    loss = loss_object(y_true=y_true, y_pred=y_pred)
    mask = tf.cast(tf.not_equal(y_true, padding_token), dtype=loss.dtype)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def inference(image, encoder, decoder, max_length, start_token=START_TOKEN, end_token=END_TOKEN, padding_token=PADDING_TOKEN):
    image = tf.expand_dims(image, 0) if len(image.shape) == 3 else tf.expand_dims(image, 0)  # Add batch dim
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    features = encoder(image, training=False)
    batch_size = tf.shape(image)[0]
    
    hidden = decoder.initialize_hidden_state(batch_size)
    
    # Prepare decoder input (could be start tokens or a fixed tensor as your decoder needs)
    dec_input = tf.expand_dims(tf.fill([batch_size], start_token), 1)  
    
    # Predict the whole sequence at once
    predictions, hidden, attention_weights = decoder(dec_input, features, hidden, training=False)
    # predictions shape: (batch_size, seq_len, vocab_size)
    
    predicted_ids = tf.argmax(predictions, axis=-1).numpy()
    
    return predicted_ids[0], attention_weights.numpy() 

def calculate_accuracy(predictions, labels, padding_token):
    total_chars = 0
    correct_chars = 0
    total_seqs = 0
    correct_seqs = 0
    
    for pred, label in zip(predictions, labels):
        # Remove padding from label
        label_no_pad = [l for l in label if l != padding_token]
        
        # Optionally, remove padding from prediction as well
        pred_no_pad = [p for p in pred if p != padding_token]
        
        # Character accuracy: count matches up to max length of label and pred
        max_len = max(len(label_no_pad), len(pred_no_pad))
        for i in range(max_len):
            l = label_no_pad[i] if i < len(label_no_pad) else None
            p = pred_no_pad[i] if i < len(pred_no_pad) else None
            if l is not None:
                total_chars += 1
                if p == l:
                    correct_chars += 1
        
        # Sequence accuracy: exact match including length and content (excluding padding)
        seq_match = (label_no_pad == pred_no_pad)
        if seq_match:
            correct_seqs += 1
        total_seqs += 1
    
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0
    
    return char_acc, seq_acc

# ============= VALIDATION FUNCTION =============
def validate(images, labels, encoder, decoder, max_length, padding_token, num_samples=None):

    if num_samples is None:
        num_samples = len(images)
    
    predictions = []
    accr = []
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        pred, _ = inference(img, encoder, decoder, max_length, 
                           start_token=0, end_token=1, padding_token=padding_token)
        predictions.append(pred)
    
    char_acc, seq_acc = calculate_accuracy(predictions[:num_samples], 
                                           labels[:num_samples], 
                                           padding_token)
    
    return accr, seq_acc, predictions

def decode_label(token_ids, remove_special=True):
    decoded = []
    
    for token_id in token_ids:
        if remove_special and token_id in [START_TOKEN, END_TOKEN, PADDING_TOKEN]:
            continue
        
        if token_id in num_to_char:
            decoded.append(num_to_char[token_id])
    
    return ''.join(decoded)

checkpoint_dir='/media/malik/ESD-USB/ChequeOCR/train/DLOCRMICRSelfTrain/checkpoints'
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    print(f"Restoring from checkpoint: {manager.latest_checkpoint}")
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
else:
    print("No checkpoint found. Cannot perform inference without pre-trained weights.")

# print("Encoder Weights:", decoder.trainable_variables)

def predict_model(test_images, encoder=encoder, decoder=decoder, max_length=MAX_LABEL_LEN, padding_token=PADDING_TOKEN):

    num_samples = len(test_images)
    
    predictions = []
    accr = []
    for i in range(min(num_samples, len(test_images))):
        img = test_images[i]
        print(img.shape)
        pred, _ = inference(img, encoder, decoder, max_length, 
                           start_token=0, end_token=1, padding_token=padding_token)
        pred = [l for l in pred if l != padding_token]
        decoded_text = decode_label(pred)
        predictions.append(decoded_text)
    return predictions
    
    # print("\nSample Predictions:")
    # for j in range(max(0, len(sample_preds))):
    #     pred = sample_preds[j]
        
    #     # Remove padding from prediction
    #     pred = [l for l in pred if l != padding_token]
    #     decoded_text = decode_label(pred)
        

if __name__=="__main__":
    def select_image_path(initial_folder="/home/green-fin/pythonfiles/Prediction/train/MICR"):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            initialdir=initial_folder,  # Set the initial directory for the dialog
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            )
        )

        if file_path:
            absolute_path = os.path.abspath(file_path)
            print(f"Selected image path: {absolute_path}")
            image = Image.open(absolute_path)
            return absolute_path#, image
        else:
            print("No file selected.")
            return None, None
    #
    path = select_image_path()
    image = preprocess_image(path)
    image = np.expand_dims(image, axis=0)
    current = time.time()
    predictions = predict_model(test_images=image)
    # predictions = predict_model(test_images=image, encoder=encoder, decoder=decoder, max_length=MAX_LABEL_LEN, padding_token=PADDING_TOKEN)
    print(f"Predictions: {predictions}")
    print(f"Running Time: {time.time()-current}")
