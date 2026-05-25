import os
import string
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from keras import regularizers
import mlflow
import mlflow.tensorflow
import logging
from datetime import datetime



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
    
def masked_accuracy(real, pred):
    # real: [batch, seq_len], pred: [batch, seq_len, vocab_size]
    mask = tf.cast(tf.math.not_equal(real, 0), tf.float32)
    pred_ids = tf.argmax(pred, axis=-1, output_type=real.dtype)  # shape: [batch, seq_len]
    correct = tf.cast(tf.equal(real, pred_ids), tf.float32)
    correct *= mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred, padding_token):
    loss = loss_object(y_true=y_true, y_pred=y_pred)
    mask = tf.cast(tf.not_equal(y_true, padding_token), dtype=loss.dtype)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

@tf.function
def train_step(tensor, target, encoder, decoder, optimizer, padding_token, teacher_forcing_ratio=0.5):
    batch_size = tf.shape(tensor)[0]
    
    with tf.GradientTape() as tape:
        features = encoder(tensor, training=False)
        hidden = decoder.initialize_hidden_state(batch_size)
        dec_input = tf.expand_dims(tf.fill([batch_size], START_TOKEN), 1)
        
        predictions, hidden, _ = decoder(dec_input, features, hidden, training=True)
        loss = masked_loss(target, predictions, padding_token)
        # loss_ = loss_fn(target, predictions)
        # mask = tf.cast(tf.not_equal(target, padding_token), dtype=loss_.dtype)
        # loss = tf.reduce_sum(loss_ * mask) / tf.reduce_sum(mask)
        
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    acc = masked_accuracy(target, predictions)
    print("Accuracy:",acc)
    return loss

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
    
    predicted_ids = tf.argmax(predictions, axis=-1).numpy()  # (batch_size, seq_len)
    
    return predicted_ids[0], attention_weights.numpy()  # return first batch sample

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
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        pred, _ = inference(img, encoder, decoder, max_length, 
                           start_token=0, end_token=1, padding_token=padding_token)
        predictions.append(pred)
    
    char_acc, seq_acc = calculate_accuracy(predictions[:num_samples], 
                                           labels[:num_samples], 
                                           padding_token)
    
    return char_acc, seq_acc, predictions



# ============= MAIN TRAINING LOOP =============
def train_model(images, labels, encoder, decoder, optimizer, 
                epochs, batch_size, padding_token, max_length,
                val_images=None, val_labels=None, checkpoint_dir='train/DLOCRMICRSelfTrain/checkpoints'):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        print(f"Restoring from checkpoint: {manager.latest_checkpoint}")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint found, starting training from scratch.")
    
    prev_acc = 93.00  # Track the previous best accuracy
    teacher_forcing_ratio = 0.8
    
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.3f}")
        print(f"{'='*60}")
        
        # Shuffle data
        num_samples = len(images)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]
        
        # Training
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_images = images_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]
            
            # Add channel dimension if needed
            if len(batch_images.shape) == 3:
                batch_images = batch_images[..., np.newaxis]
            
            # Convert to tensors
            batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
            
            # Train step
            loss = train_step(batch_images, batch_labels, encoder, decoder, 
                            optimizer, padding_token, teacher_forcing_ratio)
            
            epoch_loss += loss.numpy()
            num_batches += 1
            
            # Progress
            if (num_batches % 10) == 0:
                print(f"\033[92mBatch {num_batches}/{num_samples//batch_size} | Loss: {loss.numpy():.4f}\033[0m", end='\r')

        
        avg_loss = epoch_loss / num_batches
        print(f"\nTraining Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
        
        # Validation
        if val_images is not None and val_labels is not None:
            print("\nValidating...")
            char_acc, seq_acc, sample_preds = validate(
                val_images, val_labels, encoder, decoder, 
                max_length, padding_token, num_samples=min(100, len(val_images))
            )
            
            print(f"Validation - Char Accuracy: \033[92m{char_acc*100:.2f}%\033[0m | "
                  f"Seq Accuracy: \033[92m{seq_acc*100:.2f}%\033[0m")

            # Show sample predictions
            print("\nSample Predictions:")
            for j in range(min(3, len(sample_preds))):
                pred = sample_preds[j]
                label = [l for l in val_labels[j] if l != padding_token]
                pred = [l for l in sample_preds[j] if l != padding_token]
                print(f"  True: {label}")
                print(f"  Pred: {pred}")
                print("="*10)

            # Check if current seq_acc is better than the previous one
            if char_acc > prev_acc:
                # If improvement, save the model and update prev_acc
                print(f"New best validation accuracy! Saving model...")
                # encoder.save_weights(os.path.join(checkpoint_dir, 'best_encoder.ckpt'))
                # decoder.save_weights(os.path.join(checkpoint_dir, 'best_decoder.ckpt'))
                prev_acc = char_acc  # Update the previous accuracy with the current one
                saved_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(int(epoch), saved_path))
            else:
                # If no improvement, restore the weights from the previous best epoch
                print(f"No improvement in validation accuracy. So No Saving New weights")
                # print(f"No improvement in validation accuracy. Restoring best weights from last epoch.")
                # encoder.load_weights(os.path.join(checkpoint_dir, 'best_encoder.weights.h5'))
                # decoder.load_weights(os.path.join(checkpoint_dir, 'best_decoder.weights.h5'))

        # Decay teacher forcing
        if epoch < 10:
            teacher_forcing_ratio = 0.5
        else:
            teacher_forcing_ratio = max(0.1, teacher_forcing_ratio - CONFIG['teacher_forcing_decay'])
        
        print(f"{'='*60}")

if __name__ == "__main__":
    # Load data
    data_dir = "/home/green-fin/pythonfiles/Prediction/train/label1"
    
    images_, labels_ = prepare_data(data_dir, verbose=True)
    indices = np.random.permutation(len(images_))
    images = images_[indices]
    labels = labels_[indices]
    
    # Validate
    validate_dataset(images, labels)
#===== Defining Model===========
    encoder = OCREncoder(CONFIG)
    decoder = Decoder(CONFIG)
    encoder.build(input_shape=(None, None, None, None))  
    decoder.build(input_shape=(None, None, None))
    # MLFlow
    logging.basicConfig(level=logging.INFO)  
    logger = logging.getLogger(__name__)
    mlflow.set_experiment("Image_Captioning_Seq2Seq")
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "epochs": CONFIG['epochs'],
            "batch_size": CONFIG['batch_size'],
            "initial_lr": CONFIG['initial_lr'],
        })
    
        # Optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.90
        )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # optimizer = tf.keras.optimizers.AdamW(1e-2, weight_decay=1e-3)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)#(learning_rate=lr_schedule)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Split data into train/val (80/20)
        split_idx = int(0.8 * len(images))
        train_images, val_images = images[:split_idx], images[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        # Train
        train_model(
            images=train_images,
            labels=train_labels,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            padding_token=PADDING_TOKEN,
            max_length=MAX_LABEL_LEN,
            val_images=val_images,
            val_labels=val_labels
        )
        
        print("\nTraining complete!")
        
        # ============= TEST INFERENCE =============
        print("\nTesting inference on a sample image...")
        test_img = val_images[0]
        predicted_seq, attention = inference(
            test_img, encoder, decoder, MAX_LABEL_LEN,
            start_token=START_TOKEN, end_token=END_TOKEN, padding_token=PADDING_TOKEN
        )
        
        true_label = [l for l in val_labels[0] if l != PADDING_TOKEN]
        predicted_seq = [l for l in predicted_seq if l != PADDING_TOKEN]
        print(f"True sequence: {true_label}")
        print(f"Predicted sequence: {predicted_seq}")
        mlflow.tensorflow.log_model(tf.keras.models.Model(), artifact_path="model")
