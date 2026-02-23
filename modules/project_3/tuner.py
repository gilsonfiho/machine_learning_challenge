import sys
import os

# Garante que a raiz seja vista para os imports de modules.project_3 funcionarem
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch, BayesianOptimization
from modules.project_3.config import Config
from modules.project_3.data_loader import DataLoader

class HyperparameterTuner:
    @staticmethod
    def build_model_hp(hp):
        from tensorflow.keras.applications import EfficientNetV2S
        from tensorflow.keras import layers
        
        num_classes = len(Config.CLASSES) + 1
        
        learning_rate = hp.Choice('learning_rate', [0.00001, 0.00005, 0.0001, 0.0005])
        dropout_1 = hp.Choice('dropout_1', [0.3, 0.4, 0.5, 0.6])
        dropout_2 = hp.Choice('dropout_2', [0.2, 0.3, 0.4, 0.5])
        dense_1 = hp.Choice('dense_1', [256, 512, 1024])
        dense_2 = hp.Choice('dense_2', [128, 256, 512])
        weight_decay = hp.Choice('weight_decay', [0.00001, 0.0001, 0.001])
        
        base_model = EfficientNetV2S(
            input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_1, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(weight_decay)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_1),
            layers.Dense(dense_2, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(weight_decay)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @staticmethod
    def tune_random(X_train, y_train, X_val, y_val, max_trials=5, output_dir=None):
        search_dir = "C:/temp/hiper_results"
        if not os.path.exists("C:/temp"): os.makedirs("C:/temp", exist_ok=True)
        
        tuner = RandomSearch(
            HyperparameterTuner.build_model_hp,
            objective='val_accuracy',
            max_trials=max_trials,
            directory=search_dir,
            project_name='random_run',
            overwrite=True
        )
        
        datagen = DataLoader.augment_data(X_train)
        class_weights = Config.calculate_class_weights(y_train)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        tuner.search(
            datagen.flow(X_train, y_train, batch_size=16),
            validation_data=(X_val, y_val),
            epochs=10,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
        return tuner

    @staticmethod
    def tune_bayesian(X_train, y_train, X_val, y_val, max_trials=5, output_dir=None):
        search_dir = "C:/temp/hiper_results"
        if not os.path.exists("C:/temp"): os.makedirs("C:/temp", exist_ok=True)
        
        tuner = BayesianOptimization(
            HyperparameterTuner.build_model_hp,
            objective='val_accuracy',
            max_trials=max_trials,
            directory=search_dir,
            project_name='bayesian_run',
            overwrite=True
        )
        
        datagen = DataLoader.augment_data(X_train)
        class_weights = Config.calculate_class_weights(y_train)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        tuner.search(
            datagen.flow(X_train, y_train, batch_size=16),
            validation_data=(X_val, y_val),
            epochs=10,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
        return tuner

    @staticmethod
    def print_results(tuner, top_n=3):
        best_hps = tuner.get_best_hyperparameters(num_trials=top_n)
        for i, hp in enumerate(best_hps, 1):
            print(f"\nConfiguração {i}:")
            for param in ['learning_rate', 'dense_1', 'dense_2', 'dropout_1', 'dropout_2', 'weight_decay']:
                print(f"   {param}: {hp.get(param)}")

    @staticmethod
    def get_best_model(tuner):
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        return model, best_hps