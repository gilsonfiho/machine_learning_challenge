import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from modules.project_3.config import Config

class ModelBuilder:
    @staticmethod
    def build_model(num_classes):
        base_model = EfficientNetV2S(
            input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(Config.WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(Config.WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    @staticmethod
    def compile_model(model):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @staticmethod
    def unfreeze_base(model, base_model, num_layers=50):
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model