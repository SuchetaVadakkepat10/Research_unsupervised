"""
Configuration file for Brain MRI Classification Models
"""
import torch
import os

# ========================
# Data Configuration
# ========================
DATA_DIR = "data"  # Root directory containing class subdirectories
CLASSES = ["no_tumor", "glioblastoma", "metastasis"]
NUM_CLASSES = len(CLASSES)

# Image preprocessing
IMAGE_SIZE = 256  # Resize images to 256x256
MEAN = [0.5]  # Normalization mean for grayscale
STD = [0.5]   # Normalization std for grayscale

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ========================
# Training Configuration
# ========================
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# Model 1 (End-to-end CNN)
MODEL1_EPOCHS = 50
MODEL1_LR = 0.001
MODEL1_WEIGHT_DECAY = 1e-4

# Model 2 (VAE + MLP Classifier)
# Stage 1: VAE training
VAE_EPOCHS = 50
VAE_LR = 0.001
VAE_WEIGHT_DECAY = 1e-4
LATENT_DIM = 128
KL_WEIGHT = 0.0001  # Weight for KL divergence loss

# Stage 2: MLP Classifier training
CLASSIFIER_EPOCHS = 30
CLASSIFIER_LR = 0.001
CLASSIFIER_WEIGHT_DECAY = 1e-4

# ========================
# Model Architecture
# ========================
# CNN Model 1 architecture
CNN_CHANNELS = [1, 32, 64, 128, 256]  # Input channel followed by conv layer channels
MLP_HIDDEN_DIMS = [512, 256]  # Hidden dimensions for MLP classifier

# VAE architecture
VAE_ENCODER_CHANNELS = [1, 32, 64, 128, 256]
VAE_DECODER_CHANNELS = [256, 128, 64, 32, 1]

# MLP Classifier architecture (for Model 2)
MLP_CLASSIFIER_HIDDEN = [256, 128]

# ========================
# Output Configuration
# ========================
RESULTS_DIR = "results"
MODELS_DIR = "saved_models"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Model save paths
MODEL1_PATH = os.path.join(MODELS_DIR, "model1_cnn.pth")
VAE_PATH = os.path.join(MODELS_DIR, "model2_vae.pth")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "model2_classifier.pth")

# Results save paths
MODEL1_RESULTS = os.path.join(RESULTS_DIR, "model1_results.json")
MODEL2_RESULTS = os.path.join(RESULTS_DIR, "model2_results.json")

# ========================
# Device Configuration
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
