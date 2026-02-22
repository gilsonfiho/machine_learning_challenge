import os
import numpy as np

class Config:
    DATASET_PATH = "data/bmw10_release"
    CLASSES = [3, 4, 5]
    OTHER_CLASS = 'Outros'
    
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.00001
    WEIGHT_DECAY = 0.0001
    
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Data Augmentation
    AUGMENTATION = {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.2,
        'shear_range': 0.2,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    }
    
    # Class weights (inverso da frequência)
    @staticmethod
    def calculate_class_weights(y_train):
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        return weights
    
    MODEL_PATH = "outputs/project_3/model.h5"
    RESULTS_PATH = "outputs/project_3/results"
    
    os.makedirs(RESULTS_PATH, exist_ok=True)