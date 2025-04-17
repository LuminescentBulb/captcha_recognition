import pathlib
import string


PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "pictures"
LABEL_FILE = DATA_DIR / "labels.csv"

# --- Captcha Properties ---
IMAGE_WIDTH = 280
IMAGE_HEIGHT = 90
IMAGE_CHANNELS = 1 # Grayscale
CAPTCHA_LENGTH = 6

# Define the character set
# Alphanumeric: digits + lowercase + uppercase
CHARACTER_SET = string.digits + string.ascii_lowercase + string.ascii_uppercase
NUM_CHARACTERS = len(CHARACTER_SET) # Should be 10 + 26 + 26 = 62

# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Model Saving ---
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = "captcha_model.pth"
CHECKPOINT_FILENAME_FORMAT = "captcha_model_epoch_{epoch:02d}.pth"

# --- Add other configurations as needed ---
# e.g., validation split ratio, random seed, etc.
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation
RANDOM_SEED = 42