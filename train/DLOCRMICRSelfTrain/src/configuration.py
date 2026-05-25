import argparse
import string
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # Argument parser setup
        self.parser = argparse.ArgumentParser(description="OCR Configuration")

        # Vocabulary
        self.parser.add_argument('--char_list', type=str, default=string.digits + 'OTDB?', help="Character set (default: 0-9, A, D, O, T, space)")
        self.parser.add_argument('--start_token', type=int, default=0, help="Start token (default: 0)")
        self.parser.add_argument('--end_token', type=int, default=1, help="End token (default: 1)")
        self.parser.add_argument('--padding_token', type=int, default=2, help="Padding token (default: 2)")
        
        # Image preprocessing
        self.parser.add_argument('--img_height', type=int, default=64, help="Image height (default: 64)")
        self.parser.add_argument('--img_width', type=int, default=256, help="Image width (default: 512)")
        self.parser.add_argument('--max_label_len', type=int, default=64, help="Maximum label length (default: 64)")

        # Model architecture
        self.parser.add_argument('--conv_filters', type=int, nargs='+', default=[64, 128, 256, 512], help="Convolution filter sizes (default: [64, 128, 256, 512])")
        self.parser.add_argument('--kernel_size', type=int, nargs=2, default=(3, 3), help="Kernel size (default: (3, 3))")
        self.parser.add_argument('--embedding_dim', type=int, default=256, help="Embedding dimension (default: 256)")
        self.parser.add_argument('--decoder_units', type=int, default=128, help="Decoder units (default: 256)")
        self.parser.add_argument('--attention_units', type=int, default=512, help="Attention units (default: 512)")
        self.parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate (default: 0.3)")

        # Training
        self.parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default: 8)")
        self.parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (default: 100)")
        # self.parser.add_argument('--start_token', type=int, default=0, help="Start  Token")
        # self.parser.add_argument('--ending_token', type=int, default=1, help="Ending Token")
        # self.parser.add_argument('--padding_token', type=int, default=2, help="Padding Token")
        self.parser.add_argument('--max_len', type=int, default=64, help="Max Label Length (default: 64)")
        self.parser.add_argument('--initial_lr', type=float, default=0.001, help="Initial learning rate (default: 0.001)")
        self.parser.add_argument('--lr_decay_steps', type=int, default=1000, help="Learning rate decay steps (default: 1000)")
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.9, help="Learning rate decay rate (default: 0.9)")
        self.parser.add_argument('--gradient_clip', type=float, default=1.0, help="Gradient clipping value (default: 1.0)")
        self.parser.add_argument('--teacher_forcing_initial', type=float, default=0.8, help="Teacher forcing initial value (default: 0.8)")
        self.parser.add_argument('--teacher_forcing_decay', type=float, default=0.02, help="Teacher forcing decay rate (default: 0.02)")
        self.parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split (default: 0.2)")
        self.parser.add_argument('--early_stopping_patience', type=int, default=15, help="Early stopping patience (default: 15)")

        # Paths
        self.parser.add_argument('--data_dir', type=str, default="train/DLOCRMICRSelfTrain/data", help="Data directory (default: /home/malik/OCR/Dataset/)")
        self.parser.add_argument('--model_save_dir', type=str, default="train/DLOCRMICRSelfTrain/models", help="Model save directory (default: ./models)")
        self.parser.add_argument('--checkpoint_dir', type=str, default="train/DLOCRMICRSelfTrain/checkpoints", help="Checkpoint directory (default: ./checkpoints)")
        self.parser.add_argument('--log_dir', type=str, default="train/DLOCRMICRSelfTrain/logs", help="Log directory (default: ./logs)")

        # Parse arguments
        self.args = self.parser.parse_args()

        # Initialize configuration parameters from args
        self._initialize_config()

    def _initialize_config(self):
        # Set vocabulary
        self.CHARS = self.args.char_list
        self.START_TOKEN = self.args.start_token
        self.END_TOKEN = self.args.end_token
        self.PADDING_TOKEN = self.args.padding_token

        # Image preprocessing
        self.IMG_HEIGHT = self.args.img_height
        self.IMG_WIDTH = self.args.img_width
        self.MAX_LABEL_LEN = self.args.max_label_len

        # Model architecture
        self.CONV_FILTERS = self.args.conv_filters
        self.KERNEL_SIZE = tuple(self.args.kernel_size)  # Ensure it's a tuple
        self.EMBEDDING_DIM = self.args.embedding_dim
        self.DECODER_UNITS = self.args.decoder_units
        self.ATTENTION_UNITS = self.args.attention_units
        self.DROPOUT_RATE = self.args.dropout_rate

        # Training
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epochs
        self.INITIAL_LR = self.args.initial_lr
        self.LR_DECAY_STEPS = self.args.lr_decay_steps
        self.LR_DECAY_RATE = self.args.lr_decay_rate
        self.GRADIENT_CLIP = self.args.gradient_clip
        self.TEACHER_FORCING_INITIAL = self.args.teacher_forcing_initial
        self.TEACHER_FORCING_DECAY = self.args.teacher_forcing_decay
        self.VALIDATION_SPLIT = self.args.validation_split
        self.EARLY_STOPPING_PATIENCE = self.args.early_stopping_patience

        # Paths
        self.DATA_DIR = self.args.data_dir
        self.MODEL_SAVE_DIR = self.args.model_save_dir
        self.CHECKPOINT_DIR = self.args.checkpoint_dir
        self.LOG_DIR = self.args.log_dir

        # Create directories
        self._create_directories()

        # Build vocabulary
        self._build_vocab()

    def _create_directories(self):
        # Directories Paths
        for dir_path in [self.MODEL_SAVE_DIR, self.CHECKPOINT_DIR, self.LOG_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _build_vocab(self):
        # Vocabs 
        self.char_to_num = {char: i + 3 for i, char in enumerate(self.CHARS)}
        self.num_to_char = {i + 3: char for i, char in enumerate(self.CHARS)}
        self.num_to_char[self.START_TOKEN] = '<START>'
        self.num_to_char[self.END_TOKEN] = '<END>'
        self.num_to_char[self.PADDING_TOKEN] = '<PAD>'
        self.vocab_size = len(self.CHARS) + 3

    def to_dict(self):

        return {
            'img_height': self.IMG_HEIGHT,
            'img_width': self.IMG_WIDTH,
            'max_label_len': self.MAX_LABEL_LEN,
            'vocab_size': self.vocab_size,
            'batch_size': self.BATCH_SIZE,
            'epochs': self.EPOCHS,
            'initial_lr': self.INITIAL_LR,
            'dropout_rate': self.DROPOUT_RATE,
            'embedding_dim': self.EMBEDDING_DIM,
            'decoder_units': self.DECODER_UNITS,
        }

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")

# Usage:
if __name__ == "__main__":
    config = Config()
    # Access config arguments:
    # config.IMG_HEIGHT
    print(config.CONV_FILTERS)
    # config.char_to_num,