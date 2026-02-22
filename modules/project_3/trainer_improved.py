import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
from tensorflow import keras
from modules.project_3.config import Config

class Trainer:
    @staticmethod
    def train(model, X_train, y_train, X_val, y_val):
        # Calcular pesos das classes
        class_weights = Config.calculate_class_weights(y_train)
        
        # Data augmentation
        from modules.project_3.data_loader import DataLoader
        datagen = DataLoader.augment_data(X_train)  # ADD X_train aqui
        
        # Treinar com class weights
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            steps_per_epoch=len(X_train) // Config.BATCH_SIZE,
            class_weight=class_weights,
            verbose=1
        )
        
        model.save(Config.MODEL_PATH)
        return history
    
    @staticmethod
    def fine_tune(model, base_model, X_train, y_train, X_val, y_val):
        # Descongelar últimas camadas da base
        from modules.project_3.model_improved import ModelBuilder
        model = ModelBuilder.unfreeze_base(model, base_model, num_layers=50)
        
        # Data augmentation
        from modules.project_3.data_loader import DataLoader
        datagen = DataLoader.augment_data(X_train)
        # Calcular pesos
        class_weights = Config.calculate_class_weights(y_train)
        
        # Fine-tune
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS // 2,
            steps_per_epoch=len(X_train) // Config.BATCH_SIZE,
            class_weight=class_weights,
            verbose=1
        )
        
        model.save(Config.MODEL_PATH)
        return history
    
    @staticmethod
    def evaluate(model, X_test, y_test):
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    @staticmethod
    def predict(model, X):
        predictions = model.predict(X)
        return np.argmax(predictions, axis=1)