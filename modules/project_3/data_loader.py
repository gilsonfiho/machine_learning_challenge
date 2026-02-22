import os
import sys
import cv2
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modules.project_3.config import Config

class DataLoader:
    @staticmethod
    def load_mat_file(mat_path):
        mat_data = sio.loadmat(mat_path)
        train_indices = mat_data['train_indices'].flatten() - 1
        test_indices = mat_data['test_indices'].flatten() - 1
        annos = mat_data['annos'].flatten()
        return train_indices, test_indices, annos
    
    @staticmethod
    def split_with_mat(X, y, mat_path):
        train_indices, test_indices, annos = DataLoader.load_mat_file(mat_path)
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def normalize(X):
        return X.astype('float32') / 255.0
    
    @staticmethod
    def augment_data(X_train):
        datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        fill_mode='nearest'
        )
        return datagen
    
    @staticmethod
    def load_images_from_folder(folder_path, classes):
        images = []
        labels = []
        
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            if class_name in [str(c) for c in classes]:
                label = classes.index(int(class_name))
            else:
                label = len(classes)
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                    images.append(img)
                    labels.append(label)
                except:
                    pass
        
        return np.array(images), np.array(labels)