import numpy as np
from tensorflow import keras
from modules.project_3.config import Config
from modules.project_3.data_loader import DataLoader

class Trainer:
    @staticmethod
    def train(model, X_train, y_train, X_val, y_val):
        datagen = DataLoader.augment_data(X_train)
        
        class_weight = {
        0: 5.0,  # classe 3
        1: 5.0,  # classe 4
        2: 5.0,  # classe 5
        3: 1.0   # Outros
        }

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            steps_per_epoch=len(X_train) // Config.BATCH_SIZE,
            class_weight=class_weight,
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