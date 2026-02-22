import os

class Config:
    DATASET_PATH = "data/bmw10_release"
    CLASSES = [3, 4, 5]
    OTHER_CLASS = 'Outros'
    
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.00001
    
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    MODEL_PATH = "outputs/project_3/model.h5"
    RESULTS_PATH = "outputs/project_3/results"
    
    os.makedirs(RESULTS_PATH, exist_ok=True)