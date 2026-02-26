"""
config.py - Configurações centralizadas do projeto BMW Classifier
"""

import torch
import os

# ================== DEVICE ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== PATHS ==================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data',  'bmw10_release')
IMG_DIR = os.path.join(DATA_DIR, 'bmw10_ims')
ANNOS_FILE = 'bmw10_annos.mat'

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'project_3')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# ================== DATASET ==================
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["Outros", "Classe 1", "Classe 2", "Classe 3"]

# ================== AUGMENTATION ==================
AUGMENT_FACTOR = 10
MINORITY_CLASSES = [1, 2, 3]

# ================== TRAINING ==================
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0005

# Learning Rates
LR_TL = 1e-5
LR_FT = 1e-5
WEIGHT_DECAY = 1e-5

# LR Scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 0.000001


# ================== MODELS ==================
MODELS_AVAILABLE = ['resnet50', 'efficientnet', 'convnext']

# ================== UTILS ==================
NUM_WORKERS = 2
PIN_MEMORY = True

# ================== CONFIG DICTIONARY ==================
CONFIG = {
    'SEED': SEED,
    'DEVICE': DEVICE,
    'BATCH_SIZE': BATCH_SIZE,
    'IMG_SIZE': IMG_SIZE,
    'NUM_CLASSES': NUM_CLASSES,
    'CLASS_NAMES': CLASS_NAMES,
    'DATA_DIR': DATA_DIR,
    'IMG_DIR': IMG_DIR,
    'ANNOS_FILE': ANNOS_FILE,
    'MODELS_DIR': MODELS_DIR,
    'FIGURES_DIR': FIGURES_DIR,
    'REPORTS_DIR': REPORTS_DIR,
    'AUGMENT_FACTOR': AUGMENT_FACTOR,
    'MINORITY_CLASSES': MINORITY_CLASSES,
    'NUM_EPOCHS': NUM_EPOCHS,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'LR_TL': LR_TL,
    'LR_FT': LR_FT,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'LR_SCHEDULER_FACTOR': LR_SCHEDULER_FACTOR,
    'LR_SCHEDULER_PATIENCE': LR_SCHEDULER_PATIENCE,
    'LR_SCHEDULER_MIN_LR': LR_SCHEDULER_MIN_LR,
    'NUM_WORKERS': NUM_WORKERS,
    'PIN_MEMORY': PIN_MEMORY,
}
