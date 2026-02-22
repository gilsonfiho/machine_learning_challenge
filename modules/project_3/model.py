import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from modules.project_3.config import Config

class ModelBuilder:
    @staticmethod
    def build_model(num_classes):
        base_model = MobileNetV2(
            input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def compile_model(model):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model